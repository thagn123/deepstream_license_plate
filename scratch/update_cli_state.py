import re

# Update state.py
with open('src/lpr/state.py', 'r') as f:
    state_content = f.read()

if 'disable_laplacian' not in state_content:
    state_content += "\ndisable_laplacian = False\n"
    with open('src/lpr/state.py', 'w') as f:
        f.write(state_content)

# Update cli.py
with open('src/lpr/cli.py', 'r') as f:
    cli_content = f.read()

if '--disable-laplacian' not in cli_content:
    cli_content = cli_content.replace('disable_laplacian = False\n', '')
    cli_content = cli_content.replace('kafka_enable = False\n', 'kafka_enable = False\n    disable_laplacian = False\n')
    cli_content = cli_content.replace('elif a == "--kafka-enable":', 'elif a == "--disable-laplacian":\n            disable_laplacian = True\n        elif a == "--kafka-enable":')
    cli_content = cli_content.replace('kafka_enable=kafka_enable,', 'kafka_enable=kafka_enable,\n        disable_laplacian=disable_laplacian,')
    with open('src/lpr/cli.py', 'w') as f:
        f.write(cli_content)
