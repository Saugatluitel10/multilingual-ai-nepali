# GitHub Repository Setup Guide

## Step 1: Create Repository on GitHub

1. Go to [GitHub](https://github.com/new)
2. Fill in the repository details:
   - **Repository name**: `multilingual-ai-nepali`
   - **Description**: "Multilingual AI for Low-Resource Languages - NLP systems for Nepali-English code-mixed text"
   - **Visibility**: Choose Public or Private
   - **DO NOT** initialize with README, .gitignore, or license (we already have these)
3. Click "Create repository"

## Step 2: Connect Local Repository to GitHub

After creating the repository on GitHub, run these commands in your terminal:

```bash
cd ~/Downloads/multilingual-ai-nepali

# Add the remote repository (replace YOUR_USERNAME with your GitHub username)
git remote add origin https://github.com/YOUR_USERNAME/multilingual-ai-nepali.git

# Verify the remote was added
git remote -v

# Push your code to GitHub
git branch -M main
git push -u origin main
```

## Step 3: Verify

Visit your repository on GitHub to confirm all files have been uploaded successfully.

## Alternative: Using GitHub CLI

If you have GitHub CLI installed, you can create the repository directly from the terminal:

```bash
cd ~/Downloads/multilingual-ai-nepali

# Create repository and push (you'll be prompted to login if needed)
gh repo create multilingual-ai-nepali --public --source=. --remote=origin --push

# Or for a private repository:
gh repo create multilingual-ai-nepali --private --source=. --remote=origin --push
```

## Next Steps

After pushing to GitHub:

1. Add repository topics/tags: `machine-learning`, `nlp`, `multilingual`, `nepali`, `low-resource-languages`
2. Enable GitHub Actions if you want CI/CD
3. Add collaborators if working in a team
4. Consider adding a CONTRIBUTING.md file
5. Set up branch protection rules for the main branch

## Useful Git Commands

```bash
# Check status
git status

# Add changes
git add .

# Commit changes
git commit -m "Your commit message"

# Push changes
git push

# Pull latest changes
git pull

# Create a new branch
git checkout -b feature-name

# Switch branches
git checkout branch-name
```
