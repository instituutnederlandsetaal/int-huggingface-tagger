# Clone your repo
#git clone https://github.com/username/repo.git
#cd repo

# Create a new orphan branch (no history)
git checkout --orphan latest-commit

# Add all files
git add -A

# Commit
git commit -m "Clean history: keep only the latest commit"

# Delete the old branch (e.g., 'main' or 'master')
git branch -D master # or: git branch -D master

# Rename new branch to 'main'
git branch -m master

# Force push to GitHub
#git push --force origin main
