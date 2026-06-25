/* ── State ─────────────────────────────────────────────────────────────────── */
const state = {
  paused: false,
  buffer: [],
  devices: {},
  filters: {},
  feedCount: 0,
  MAX_CARDS: 200,
};

/* ── Clock ─────────────────────────────────────────────────────────────────── */
function updateClock() {
  const now = new Date();
  document.getElementById('clock').textContent =
    now.toLocaleString('vi-VN', { hour12: false, timeZone: 'Asia/Ho_Chi_Minh' });
}
setInterval(updateClock, 1000);
updateClock();

/* ── WebSocket ─────────────────────────────────────────────────────────────── */
let ws = null;
let wsReconnectTimer = null;

function connectWS() {
  const proto = location.protocol === 'https:' ? 'wss' : 'ws';
  ws = new WebSocket(`${proto}://${location.host}/ws`);

  ws.onopen = () => {
    setWsDot('connected', 'Live');
    clearTimeout(wsReconnectTimer);
  };

  ws.onclose = () => {
    setWsDot('disconnected', 'Reconnecting…');
    wsReconnectTimer = setTimeout(connectWS, 3000);
  };

  ws.onerror = () => {
    ws.close();
  };

  ws.onmessage = (evt) => {
    try {
      const msg = JSON.parse(evt.data);
      handleWsMessage(msg);
    } catch (_) {}
  };
}

function setWsDot(cls, label) {
  const dot = document.getElementById('ws-dot');
  const lbl = document.getElementById('ws-label');
  dot.className = cls;
  lbl.textContent = label;
}

function handleWsMessage(msg) {
  if (msg.type === 'event_created') {
    receiveEvent(msg.data);
  } else if (msg.type === 'device_heartbeat') {
    updateDevice(msg.data);
  } else if (msg.type === 'device_offline') {
    markDeviceOffline(msg.data);
  }
}

/* ── Devices ───────────────────────────────────────────────────────────────── */
function updateDevice(data) {
  const key = `${data.device_id}|${data.project_name}`;
  state.devices[key] = { ...state.devices[key], ...data };
  renderDeviceList();
  updateHeaderStats();
}

function markDeviceOffline(data) {
  const key = `${data.device_id}|${data.project_name}`;
  if (state.devices[key]) {
    state.devices[key].status = 'offline';
    renderDeviceList();
    updateHeaderStats();
  }
}

function renderDeviceList() {
  const list = document.getElementById('device-list');
  const devs = Object.values(state.devices);

  if (!devs.length) {
    list.innerHTML = `<div style="padding:20px; text-align:center; color:var(--text-dim); font-size:12px;">Waiting for devices…</div>`;
    return;
  }

  devs.sort((a, b) => {
    if (a.status === 'online' && b.status !== 'online') return -1;
    if (b.status === 'online' && a.status !== 'online') return 1;
    return 0;
  });

  list.innerHTML = devs.map(d => {
    const online = d.status === 'online';
    const tempClass = d.gpu_temp >= 75 ? 'hot' : d.gpu_temp >= 60 ? 'warm' : 'cool';
    return `
      <div class="device-card">
        <div class="device-top">
          <div class="device-name" title="${escHtml(d.hostname || d.device_id)}">${escHtml(d.hostname || d.device_id)}</div>
          <div class="status-dot ${online ? 'online' : 'offline'}" title="${online ? 'Online' : 'Offline'}"></div>
        </div>
        <div class="device-meta">
          <div class="row"><span class="key">Project</span><span>${escHtml(d.project_name || '—')}</span></div>
          <div class="row"><span class="key">IP</span><span>${escHtml(d.ip_address || '—')}</span></div>
          <div class="row">
            <span class="key">FPS</span>
            <span>${d.fps != null ? d.fps.toFixed(1) : '—'}</span>
          </div>
          <div class="row">
            <span class="key">GPU Temp</span>
            <span class="temp-indicator ${tempClass}">${d.gpu_temp != null ? d.gpu_temp.toFixed(1) + '°C' : '—'}</span>
          </div>
        </div>
      </div>`;
  }).join('');
}

/* ── Events / Feed ─────────────────────────────────────────────────────────── */
function receiveEvent(data) {
  if (!matchesFilters(data)) return;

  if (state.paused) {
    state.buffer.push(data);
    document.getElementById('pause-banner').textContent =
      `⏸ Feed paused — ${state.buffer.length} events buffered`;
    return;
  }

  prependCard(data);
  hideFeedEmpty();
}

function prependCard(data) {
  const feed = document.getElementById('feed');
  const card = buildCard(data);
  feed.insertBefore(card, feed.firstChild);

  state.feedCount++;

  const cards = feed.querySelectorAll('.event-card');
  if (cards.length > state.MAX_CARDS) {
    cards[cards.length - 1].remove();
  }
}

function hideFeedEmpty() {
  document.getElementById('feed-empty').style.display = 'none';
}

function buildCard(data) {
  const el = document.createElement('div');
  const typeClass = eventTypeClass(data.event_type);
  el.className = `event-card ${typeClass}`;
  el.dataset.eventId = data.event_id;

  const time = data.timestamp ? fmtTime(data.timestamp) : '—';
  const badgeHtml = `<span class="badge ${typeClass}">${fmtType(data.event_type)}</span>`;
  const thumbSrc = bestThumb(data.images);
  const thumbHtml = thumbSrc
    ? `<img class="card-thumb" src="${thumbSrc}" alt="thumb" loading="lazy" onerror="this.parentElement.innerHTML='<div class=no-thumb>🖼</div>'" />`
    : `<div class="no-thumb">🖼</div>`;

  el.innerHTML = `
    <div class="card-left">
      <div class="card-header">
        ${badgeHtml}
        <span class="card-device">${escHtml(data.device_id)} › ${escHtml(data.camera_name || data.camera_id || '—')}</span>
        <span class="card-time">${time}</span>
      </div>
      <div class="card-message">${escHtml(data.message || '')}</div>
      ${data.plate_text ? `<div class="plate-badge">${escHtml(data.plate_text)}</div>` : ''}
    </div>
    <div class="card-right">${thumbHtml}</div>`;

  el.addEventListener('click', () => openDrawer(data.event_id));
  return el;
}

function bestThumb(images) {
  if (!images) return null;
  const order = ['plate', 'vehicle', 'frame'];
  for (const k of order) {
    const url = images[`${k}_image_url`];
    if (url) return url;
  }
  return null;
}

/* ── Pause / Resume / Clear ────────────────────────────────────────────────── */
function togglePause() {
  state.paused = !state.paused;
  const btn = document.getElementById('btn-pause');

  if (state.paused) {
    btn.textContent = '▶ Resume';
    btn.classList.add('btn-pause-active');
    document.getElementById('pause-banner').classList.add('visible');
    document.getElementById('pause-banner').textContent = '⏸ Feed paused — 0 events buffered';
  } else {
    btn.textContent = '⏸ Pause';
    btn.classList.remove('btn-pause-active');
    document.getElementById('pause-banner').classList.remove('visible');

    if (state.buffer.length) {
      state.buffer.forEach(d => prependCard(d));
      state.buffer = [];
      hideFeedEmpty();
    }
  }
}

function clearFeed() {
  document.getElementById('feed').innerHTML = '';
  state.feedCount = 0;
  state.buffer = [];
  document.getElementById('feed-empty').style.display = '';
}

/* ── Filters ───────────────────────────────────────────────────────────────── */
function applyFilters() {
  state.filters = {
    project: document.getElementById('f-project').value.trim().toLowerCase(),
    device:  document.getElementById('f-device').value.trim().toLowerCase(),
    plate:   document.getElementById('f-plate').value.trim().toLowerCase(),
    type:    document.getElementById('f-type').value,
  };
  loadHistory();
}

function clearFilters() {
  document.getElementById('f-project').value = '';
  document.getElementById('f-device').value = '';
  document.getElementById('f-plate').value = '';
  document.getElementById('f-type').value = '';
  state.filters = {};
  loadHistory();
}

function matchesFilters(data) {
  const f = state.filters;
  if (f.project && !data.project_name?.toLowerCase().includes(f.project)) return false;
  if (f.device  && !data.device_id?.toLowerCase().includes(f.device)) return false;
  if (f.plate   && !data.plate_text?.toLowerCase().includes(f.plate)) return false;
  if (f.type    && data.event_type !== f.type) return false;
  return true;
}

/* ── Load initial history from API ────────────────────────────────────────── */
async function loadHistory() {
  clearFeed();
  const params = new URLSearchParams({ limit: 80 });
  const f = state.filters;
  if (f.project) params.set('project_name', f.project);
  if (f.device)  params.set('device_id',    f.device);
  if (f.plate)   params.set('plate_text',    f.plate);
  if (f.type)    params.set('event_type',    f.type);

  try {
    const res = await fetch(`/api/events?${params}`);
    if (!res.ok) return;
    const json = await res.json();
    const items = (json.items || []).reverse();
    if (items.length) {
      hideFeedEmpty();
      items.forEach(ev => {
        const imgs = {};
        (ev.images || []).forEach(i => {
          imgs[`${i.image_type}_image_url`] = i.is_available ? i.thumb_url : null;
        });
        prependCard({ ...ev, images: imgs });
      });
    }
  } catch (e) {
    console.warn('loadHistory error', e);
  }
}

async function loadDevices() {
  try {
    const res = await fetch('/api/devices');
    if (!res.ok) return;
    const devs = await res.json();
    devs.forEach(d => {
      const key = `${d.device_id}|${d.project_name}`;
      state.devices[key] = d;
    });
    renderDeviceList();
  } catch (e) {}
}

async function loadStats() {
  try {
    const res = await fetch('/api/stats');
    if (!res.ok) return;
    const s = await res.json();
    document.getElementById('stat-total').textContent   = s.total_events ?? '—';
    document.getElementById('stat-online').textContent  = s.online_devices ?? '—';
    document.getElementById('stat-devices').textContent = s.total_devices ?? '—';
  } catch (e) {}
}

function updateHeaderStats() {
  const devs = Object.values(state.devices);
  const online = devs.filter(d => d.status === 'online').length;
  document.getElementById('stat-online').textContent  = online;
  document.getElementById('stat-devices').textContent = devs.length;
}

/* ── Detail Drawer ─────────────────────────────────────────────────────────── */
async function openDrawer(eventId) {
  const drawer = document.getElementById('drawer');
  const body   = document.getElementById('drawer-body');
  body.innerHTML = `<div style="text-align:center; color:var(--text-dim); padding:40px; font-size:13px;">Đang tải…</div>`;
  drawer.classList.add('open');
  document.getElementById('overlay').classList.add('open');

  try {
    const res = await fetch(`/api/events/${encodeURIComponent(eventId)}`);
    if (!res.ok) { body.innerHTML = '<p style="color:var(--rose)">Không tìm thấy sự kiện.</p>'; return; }
    const ev = await res.json();
    body.innerHTML = buildDrawerContent(ev);
    setupImageTabs(body, ev.images || []);
  } catch (e) {
    body.innerHTML = `<p style="color:var(--rose)">Lỗi: ${e.message}</p>`;
  }
}

function buildDrawerContent(ev) {
  const images = ev.images || [];

  const tabsHtml = images.length
    ? `<div class="image-tabs">${images.map((img, i) =>
        `<button class="img-tab ${i === 0 ? 'active' : ''}" onclick="switchTab(this, ${i})" data-idx="${i}">${escHtml(img.image_type)}</button>`
      ).join('')}</div>`
    : '';

  const firstImg = images[0];
  const imgHtml = firstImg
    ? (firstImg.is_available
        ? `<img id="drawer-main-img" src="${escHtml(firstImg.image_url || firstImg.thumb_url)}" alt="${escHtml(firstImg.image_type)}" />`
        : `<div class="image-unavailable"><span>🗑️</span><span>Ảnh gốc đã được dọn dẹp tự động để tiết kiệm dung lượng</span></div>`)
    : `<div class="image-unavailable"><span>🖼</span><span>Không có ảnh</span></div>`;

  const rows = [
    ['Event ID',    ev.event_id,    false],
    ['Project',     ev.project_name, false],
    ['Device',      ev.device_id,    false],
    ['Camera',      `${ev.camera_id || ''} ${ev.camera_name ? '(' + ev.camera_name + ')' : ''}`.trim(), false],
    ['Event Type',  ev.event_type,   false],
    ['Plate',       ev.plate_text,   true],
    ['Confidence',  ev.confidence != null ? (ev.confidence * 100).toFixed(1) + '%' : null, false],
    ['Track ID',    ev.track_id,     false],
    ['Object ID',   ev.object_id,    false],
    ['BBox',        ev.bbox,         false],
    ['FPS',         ev.fps != null ? ev.fps.toFixed(1) : null, false],
    ['Model',       ev.model_name,   false],
    ['Model Ver',   ev.model_version, false],
    ['Timestamp',   ev.timestamp ? fmtTime(ev.timestamp) : null, false],
  ].filter(r => r[1] != null && r[1] !== '');

  const metaHtml = rows.map(([k, v, isPlate]) =>
    `<div class="meta-row">
      <div class="meta-key">${escHtml(k)}</div>
      <div class="meta-val ${isPlate ? 'plate' : ''}">${escHtml(String(v))}</div>
    </div>`
  ).join('');

  let rawHtml = '';
  if (ev.raw_metadata) {
    let prettyJson = ev.raw_metadata;
    try { prettyJson = JSON.stringify(JSON.parse(ev.raw_metadata), null, 2); } catch(_) {}
    rawHtml = `
      <div class="meta-section-title">Raw Metadata (DeepStream payload)</div>
      <div class="raw-json-block"><pre>${escHtml(prettyJson)}</pre></div>`;
  }

  return `
    ${tabsHtml}
    <div class="drawer-image-container" id="drawer-img-container">${imgHtml}</div>
    <div class="meta-section-title">Event Metadata</div>
    <div class="meta-table">${metaHtml}</div>
    ${rawHtml}`;
}

function setupImageTabs(body, images) {
  body._images = images;
}

function switchTab(btn, idx) {
  const body = document.getElementById('drawer-body');
  const images = body._images || [];
  const img = images[idx];
  if (!img) return;

  body.querySelectorAll('.img-tab').forEach(t => t.classList.remove('active'));
  btn.classList.add('active');

  const container = document.getElementById('drawer-img-container');
  if (img.is_available) {
    container.innerHTML = `<img id="drawer-main-img" src="${escHtml(img.image_url || img.thumb_url)}" alt="${escHtml(img.image_type)}" />`;
  } else {
    container.innerHTML = `<div class="image-unavailable"><span>🗑️</span><span>Ảnh gốc đã được dọn dẹp tự động để tiết kiệm dung lượng</span></div>`;
  }
}

function closeDrawer() {
  document.getElementById('drawer').classList.remove('open');
  document.getElementById('overlay').classList.remove('open');
}

/* ── Helpers ───────────────────────────────────────────────────────────────── */
function eventTypeClass(t) {
  if (t === 'license_plate_detected') return 'lpr';
  if (t === 'vehicle_detected') return 'vehicle';
  if (t === 'error') return 'error';
  return 'default';
}

function fmtType(t) {
  if (t === 'license_plate_detected') return 'LPR';
  if (t === 'vehicle_detected') return 'Vehicle';
  if (t === 'error') return 'Error';
  return t || 'Event';
}

function fmtTime(isoStr) {
  try {
    return new Date(isoStr).toLocaleString('vi-VN', {
      hour12: false,
      timeZone: 'Asia/Ho_Chi_Minh',
      year: 'numeric', month: '2-digit', day: '2-digit',
      hour: '2-digit', minute: '2-digit', second: '2-digit',
    });
  } catch (_) { return isoStr; }
}

function escHtml(str) {
  if (str == null) return '';
  return String(str)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');
}

/* ── Init ──────────────────────────────────────────────────────────────────── */
async function init() {
  await loadDevices();
  await loadStats();
  await loadHistory();
  connectWS();

  // Refresh stats every 30s
  setInterval(loadStats, 30000);
  setInterval(loadDevices, 30000);
}

init();
