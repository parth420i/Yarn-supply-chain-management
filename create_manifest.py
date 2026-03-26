import json
import os

deploy_dir = 'deploy_temp'

# Correct template index.html content (from template_info.json)
template_index = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>APP_TITLE</title>
    <link rel="stylesheet" href="./src/styles.css">
</head>
<body>
    <div id="app">APP_CONTENT</div>
    <script type="module" src="./src/main.ts"></script>
</body>
</html>"""

# My desired index.html content
with open(os.path.join(deploy_dir, 'index.html'), 'r', encoding='utf-8') as f:
    target_index = f.read()

# Files to include in the upload manifest
files_to_upload = [
    ('public/app.py', 'public/app.py'),
    ('public/main.py', 'public/main.py'),
    ('public/yarn_supplychain_surat.csv', 'public/yarn_supplychain_surat.csv')
]

manifest = {"files": []}

# Add index.html as a DIFF to satisfy validation rules
manifest["files"].append({
    "filename": "index.html",
    "diffs": [{
        "from": template_index,
        "to": target_index
    }]
})

# Add other new files as CONTENT
for local_path, repo_path in files_to_upload:
    full_path = os.path.join(deploy_dir, local_path)
    with open(full_path, 'r', encoding='utf-8') as f:
        content = f.read()
        manifest["files"].append({
            "filename": repo_path,
            "content": content
        })

with open('deploy_temp/manifest.json', 'w', encoding='utf-8') as f:
    json.dump(manifest, f)
