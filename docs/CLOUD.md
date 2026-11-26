# Using Claude Code in the cloud with this repo

**What a cloud session can and cannot do here.** A cloud session is a fresh Ubuntu VM (4 vCPU, 16 GB RAM,
30 GB disk, **no GPU**) with your GitHub repo cloned into it. It cannot log in to ORCD, because ORCD
needs Duo and cloud sessions have no interactive login.

So the workflow is a loop:

```
cloud session: verify / change code ──push──► GitHub ──git pull──► ORCD: train ──push results──► cloud: analyse
```

The real training always runs on ORCD (see [ORCD.md](ORCD.md)).

Official docs: https://code.claude.com/docs/en/web-quickstart.md and
https://code.claude.com/docs/en/cloud-environments.md

## 1. Put the repo on GitHub (once, from your laptop)

```bash
cd reproduction_zachswork
git add -A
git commit -m "Reproduction pipeline for EDL / residual specialists / causal router"
gh repo create moe-repro --private --source . --push      # or create it on github.com and `git push`
```

`work/`, `cache/` and `.venv/` are git-ignored, so only code and docs are pushed.

## 2. Connect Claude Code on the web (once)

1. Go to https://claude.ai/code, sign in with your claude.ai account, and sign in with GitHub when
   prompted. Cloud sessions need a Pro, Max, Ultra, Team or eligible Enterprise seat; check what your
   plan includes.
2. Because the repo is **private**, install the Claude GitHub App on your account
   (https://github.com/apps/claude) and give it access to `moe-repro`.
3. Configure the environment (the environment selector under the message box → settings icon):
   - **Network access:** keep **Limited** with "Allow package managers" on, and add under
     *Allowed domains*:
     ```
     download.pytorch.org
     download-r2.pytorch.org
     huggingface.co
     *.huggingface.co
     *.hf.co
     openaipublic.blob.core.windows.net
     ```
     The first two are needed for the CPU torch build (download.pytorch.org redirects to
     download-r2.pytorch.org). Without them the setup script falls back to the PyPI wheel, which
     works but is several GB larger. The others are only needed if you want the
     session to test the real-data path (M2D2, e5, GPT-2 BPE).
   - **Setup script:** paste the contents of [`scripts/cloud_setup.sh`](../scripts/cloud_setup.sh). It
     installs CPU torch and the requirements, and is cached for later sessions.

## 3. Start a session

**From the browser:** choose `moe-repro` in the repository selector, pick the permission mode
**Auto** (or **Plan** if you want to approve a plan first), paste a prompt from
[CLOUD_PROMPT.md](../CLOUD_PROMPT.md) (start with Prompt 1), and press Enter.

**From your terminal:** push first, because the VM clones GitHub rather than your local folder.

```bash
claude --cloud "$(./scripts/cloud_prompt.sh 1)"
```

`scripts/cloud_prompt.sh N` prints prompt N from CLOUD_PROMPT.md from any directory, and fails with
an error rather than sending an empty prompt.

Each `--cloud` call starts an independent parallel session. Follow up with
`claude -p "message" --cloud <session-id>`.

The repo's `CLAUDE.md` is loaded automatically. It tells the session to stay on the smoke profile
and to run the tests before committing.

## 4. Get the work back

- When the session stops, it pushes a **branch**. Open the **Changes** card to review the diff, then
  click **Create PR** if you want one. It never opens a PR by itself.
- To continue the conversation locally: `claude --teleport` (pick the session) from a clean checkout
  of the same repo. This is one-way; local work does not sync back to the cloud session.
- Merge the branch, then on ORCD: `git pull && ./slurm/submit_all.sh`.

## 5. Analyse ORCD results in the cloud

On ORCD, after the report job has finished:

```bash
mkdir -p results_orcd
rsync -av --include='*/' --include='*.json' --include='REPORT.md' --exclude='*' work/ results_orcd/
git add results_orcd && git commit -m "ORCD results" && git push
```

Then start a cloud session with **Prompt 3** from CLOUD_PROMPT.md.

## Limits to keep in mind

- A command can run for at most about 10 minutes in the foreground and about 30 more in the
  background. The smoke pipeline takes about 2 minutes on 4 vCPUs, so it fits easily.
- An idle session pauses after a few minutes. Running commands are not restored when it resumes,
  but the conversation is.
- Cloud usage counts toward your plan's limits.
