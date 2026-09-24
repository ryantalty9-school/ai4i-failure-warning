#!/bin/bash
# Step 8: create the tests + CI branch, run the tests, and push the branch so you can open the pull request.
# Run this AFTER 7_push_to_github.command and after you have created the issue on GitHub.
cd "$(dirname "$0")" || exit 1
mkdir -p logs
exec > >(tee logs/8_open_pr.log) 2>&1
source scripts/common.sh
BRANCH="add-tests-and-ci"

if ! git remote | grep -q '^origin$'; then
  echo "ERROR: no GitHub remote yet. Run 7_push_to_github.command first."
  read -r -p "Press Return to close..."; exit 1
fi
read -r -p "Issue number you created on GitHub (just the number, e.g. 1): " N
[ -z "$N" ] && { echo "No issue number entered."; read -r -p "Press Return..."; exit 1; }

step "1/5 BRANCH"
git checkout main >/dev/null 2>&1
git checkout -b "$BRANCH" 2>/dev/null || git checkout "$BRANCH"
echo "on branch $(git rev-parse --abbrev-ref HEAD)"

step "2/5 ADD THE TEST FILES"
cp -R _pr_files/tests .
mkdir -p .github/workflows && cp _pr_files/ci_workflow.yml .github/workflows/ci.yml
cp _pr_files/requirements-dev.txt _pr_files/pytest.ini .
python - <<'PY'
import pathlib
readme = pathlib.Path("README.md"); text = readme.read_text()
section = pathlib.Path("_pr_files/README_tests_section.md").read_text()
if "## Tests" not in text:
    readme.write_text(text.replace("## Author\n", section + "## Author\n", 1))
    print("README: added the Tests section")
else:
    print("README: Tests section already there")
PY
ls tests .github/workflows

step "3/5 RUN THE TESTS"
python -m pip install -q -r requirements-dev.txt
python -m pytest || { echo "TESTS FAILED - fix them before opening the PR"; finish "FAILED"; exit 1; }

step "4/5 COMMIT AND PUSH"
git add tests .github requirements-dev.txt pytest.ini README.md
git commit -q -m "Add pytest suite and GitHub Actions CI (closes #$N)" && echo "commit created" || echo "(nothing new to commit)"
git push -u origin "$BRANCH" || { echo "push failed - check the message above"; finish "FAILED"; exit 1; }

step "5/5 OPEN THE PULL REQUEST"
URL=$(git remote get-url origin | sed -e 's/\.git$//' -e 's#git@github.com:#https://github.com/#')
sed "s/#ISSUE_NUMBER/#$N/" _pr_files/PR_BODY_TEMPLATE.md > _pr_files/PR_BODY.md
command -v pbcopy >/dev/null && pbcopy < _pr_files/PR_BODY.md && echo "The PR description is on your clipboard - paste it into the PR body."
echo
echo "Open this link, then click 'Create pull request':"
echo "  $URL/compare/$BRANCH?expand=1"
echo "Title: Add pytest suite and GitHub Actions CI"
open "$URL/compare/$BRANCH?expand=1" 2>/dev/null
finish "BRANCH PUSHED - open the link above to create the PR"
