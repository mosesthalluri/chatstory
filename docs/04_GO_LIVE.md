# Going live — making your laptop reachable from a phone

Once ChatBook works locally on your laptop, you'll want it reachable
from anywhere — your phone, your friend's phone, a tester's browser.

You have two options. Cloudflare Tunnel is what I recommend.

## Option A: Cloudflare Tunnel (recommended for v0.1)

Cloudflare Tunnel gives you a public HTTPS URL pointing to your laptop,
through Cloudflare's network. No port forwarding, no router config, no
firewall changes. Free for personal use.

### Setup (one-time, ~10 minutes)

1. **Install cloudflared.**

   On Windows: download `cloudflared-windows-amd64.exe` from
   https://github.com/cloudflare/cloudflared/releases. Rename it to
   `cloudflared.exe` and put it somewhere on your PATH (e.g.
   `C:\Windows\System32`).

   On Mac: `brew install cloudflared`

   On Linux: see https://pkg.cloudflare.com/

2. **Verify it works:** open a terminal and run `cloudflared --version`.

### Run the tunnel

With ChatBook running on http://localhost:8000, open a *second* terminal
and run:

```bash
cloudflared tunnel --url http://localhost:8000
```

After a few seconds you'll see something like:

```
Your quick Tunnel has been created! Visit it at (it may take some time
to be reachable):
https://random-words-12345.trycloudflare.com
```

That URL works from anywhere. Test it on your phone — it should show
the ChatBook upload page over HTTPS.

The URL changes every time you restart the tunnel, which is fine for
testing. For a permanent URL, see Cloudflare's docs on named tunnels.

### Limitations

- **Your laptop has to be on and running both ChatBook and cloudflared.**
  If you close the lid, the URL stops working.
- **Cloudflare's free tier has bandwidth limits.** For a few users
  uploading chats, you'll never hit them. For viral traffic, you'd hit
  them fast — but by then you should be on a real server anyway.
- **Long-running pipelines tie up your laptop.** A 200k-message chat
  takes 15-30 minutes to process and one user at a time on Ollama.

## Option B: Cheap cloud VM (when you have paying users)

When you're handling 5+ books per day, move ChatBook to a real server.

Recommended starter: **Oracle Cloud Free Tier ARM VM** (4 cores, 24GB
RAM, free forever). Real server, real uptime, no laptop needed.

Other reasonable choices: DigitalOcean ($6/month), Hetzner (€4/month),
Railway/Render ($5-10/month with managed deployment).

I'm not going to write that guide here because it depends on your
choice. The general process:

1. Get a Linux VM with Python 3.11+
2. Install ChatBook the same way you did locally (clone repo, pip install)
3. Run uvicorn behind Caddy or Nginx (handles HTTPS automatically)
4. Use systemd to auto-restart on boot

ChatGPT or Claude can walk you through this step-by-step for whatever
provider you pick.

## Domain name (optional)

If you want `mychatbook.com` instead of `random-words.trycloudflare.com`:

1. Buy a domain on Namecheap, Porkbun, or Cloudflare Registrar (~$10/year)
2. In Cloudflare, set up a "Named Tunnel" pointing to your domain
3. Done

Real domains aren't required for testing — the random Cloudflare URLs
work fine to get feedback from your first users.
