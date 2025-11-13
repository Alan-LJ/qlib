with open('MFT_Project/configs/workflow_config_multi_modal.yaml', 'rb') as f:
    content = f.read()
with open('MFT_Project/configs/workflow_config_multi_modal.yaml', 'w', encoding='utf-8') as f:
    f.write(content.decode('utf-8', errors='ignore'))