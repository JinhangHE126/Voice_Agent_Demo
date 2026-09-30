# EC2 + Cloudflare Tunnel 演示啟動說明

本文件記錄如何喺 Amazon EC2 上啟動語音客服 Demo，並用 Cloudflare Tunnel 產生 `https://...trycloudflare.com` 連結，方便手機 Chrome 測試麥克風。

注意：

- 唔好把 API Key / `.env` 機密内容寫入本文件。
- 演示期間要同時保持兩個進程：語音服務 + Cloudflare Tunnel。
- 公網 `http://IP:7862` 通常唔畀麥克風；手機演示請用 HTTPS 隧道連結。

---

## 粤语版本

### 1. 開終端 A：登入 EC2 並啟動語音服務

喺本機 PowerShell：

```powershell
ssh -i "C:\Users\pg9\Desktop\EproVoiceAgentDemo-key.pem" ubuntu@3.26.230.195
```

登入成功後：

```bash
cd ~/voice-agent-demo
source .venv/bin/activate
python scripts/run_web.py
```

見到類似下面呢句就得：

```text
Uvicorn running on http://0.0.0.0:7862
```

呢個視窗唔好關。

### 2. 開終端 B：再登入並啟動 Cloudflare Tunnel

再開一個 PowerShell：

```powershell
ssh -i "C:\Users\pg9\Desktop\EproVoiceAgentDemo-key.pem" ubuntu@3.26.230.195
```

然後：

```bash
cloudflared tunnel --url http://127.0.0.1:7862
```

會見到類似：

```text
https://xxxxx.trycloudflare.com
```

複製呢個 https 連結。呢個視窗都唔好關。

### 3. 手機測試

1. 用手機 Chrome 打開上面嘅 `https://xxxxx.trycloudflare.com`
2. 撳「開始聽筒」
3. 允許麥克風權限
4. 開始對話測試

### 4. 常見問題（粤语）

| 情況 | 原因 | 處理 |
|---|---|---|
| 502 Bad gateway | 語音服務無開，或者 Tunnel 指錯埠 | 確認 `python scripts/run_web.py` 喺跑，再用 `curl -I http://127.0.0.1:7862` 檢查 |
| 公網 `http://IP:7862` 開到頁但聽唔到 | 瀏覽器禁止非 HTTPS 麥克風 | 改用 Cloudflare HTTPS 連結 |
| 連唔上 SSH | `.pem` 路徑錯，或者 Key 唔啱 | 確認密钥文件係桌面 `EproVoiceAgentDemo-key.pem` |
| 關掉任一視窗後失效 | 服務或隧道斷咗 | 重新開終端 A 同終端 B |

### 5. 首次部署補充（粤语）

如果係全新 EC2，先要：

1. 開安全組：`22`（SSH）、`7862`（測試用）
2. 上傳项目：

```powershell
scp -i "C:\Users\pg9\Desktop\EproVoiceAgentDemo-key.pem" -r "C:\Users\pg9\Desktop\Epro\voice-agent-demo" "C:\Users\pg9\Desktop\Epro\repair-order-agent" ubuntu@3.26.230.195:~/
```

3. 伺服器安裝依賴：

```bash
sudo apt update
sudo apt install -y python3 python3-pip python3-venv
cd ~/voice-agent-demo
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

4. `.env` 入面設定：

```env
HOST=0.0.0.0
```

5. 安裝 cloudflared（如未安裝）：

```bash
curl -L --output cloudflared.deb https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64.deb
sudo dpkg -i cloudflared.deb
cloudflared --version
```

---

## 中文版本

### 1. 打开终端 A：登录 EC2 并启动语音服务

在本机 PowerShell：

```powershell
ssh -i "C:\Users\pg9\Desktop\EproVoiceAgentDemo-key.pem" ubuntu@3.26.230.195
```

登录成功后：

```bash
cd ~/voice-agent-demo
source .venv/bin/activate
python scripts/run_web.py
```

看到类似下面这行即可：

```text
Uvicorn running on http://0.0.0.0:7862
```

这个窗口不要关闭。

### 2. 打开终端 B：再登录并启动 Cloudflare Tunnel

再开一个 PowerShell：

```powershell
ssh -i "C:\Users\pg9\Desktop\EproVoiceAgentDemo-key.pem" ubuntu@3.26.230.195
```

然后执行：

```bash
cloudflared tunnel --url http://127.0.0.1:7862
```

会出现类似：

```text
https://xxxxx.trycloudflare.com
```

复制这个 https 链接。这个窗口也不要关闭。

### 3. 手机测试

1. 用手机 Chrome 打开上面的 `https://xxxxx.trycloudflare.com`
2. 点击「开始听筒」
3. 允许麦克风权限
4. 开始对话测试

### 4. 常见问题（中文）

| 情况 | 原因 | 处理 |
|---|---|---|
| 502 Bad gateway | 语音服务没启动，或隧道端口不对 | 确认 `python scripts/run_web.py` 正在运行，并用 `curl -I http://127.0.0.1:7862` 检查 |
| 公网 `http://IP:7862` 能打开页面但没有麦克风 | 浏览器禁止非 HTTPS 使用麦克风 | 改用 Cloudflare HTTPS 链接 |
| SSH 登录失败 | `.pem` 路径错误，或密钥不匹配 | 确认密钥文件是桌面上的 `EproVoiceAgentDemo-key.pem` |
| 关闭任一窗口后失效 | 服务或隧道中断 | 重新打开终端 A 和终端 B |

### 5. 首次部署补充（中文）

如果是全新 EC2，需要先：

1. 安全组放行：`22`（SSH）、`7862`（测试用）
2. 上传项目：

```powershell
scp -i "C:\Users\pg9\Desktop\EproVoiceAgentDemo-key.pem" -r "C:\Users\pg9\Desktop\Epro\voice-agent-demo" "C:\Users\pg9\Desktop\Epro\repair-order-agent" ubuntu@3.26.230.195:~/
```

3. 服务器安装依赖：

```bash
sudo apt update
sudo apt install -y python3 python3-pip python3-venv
cd ~/voice-agent-demo
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

4. 在 `.env` 中设置：

```env
HOST=0.0.0.0
```

5. 安装 cloudflared（如未安装）：

```bash
curl -L --output cloudflared.deb https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64.deb
sudo dpkg -i cloudflared.deb
cloudflared --version
```

### 6. 给老板/客户的说明（中文）

- EC2 已可部署本项目的实时语音客服。
- 电脑本地隧道或 Cloudflare HTTPS 隧道都可测试。
- 客户手机演示必须使用 HTTPS 链接，否则浏览器会禁用麦克风。
- 当前 `trycloudflare.com` 适合临时演示；长期正式环境建议改用固定域名 + HTTPS。

---

## English Version

### 1. Terminal A: SSH into EC2 and start the voice service

On your local PowerShell:

```powershell
ssh -i "C:\Users\pg9\Desktop\EproVoiceAgentDemo-key.pem" ubuntu@3.26.230.195
```

After login:

```bash
cd ~/voice-agent-demo
source .venv/bin/activate
python scripts/run_web.py
```

You should see something like:

```text
Uvicorn running on http://0.0.0.0:7862
```

Keep this terminal open.

### 2. Terminal B: SSH again and start Cloudflare Tunnel

Open another PowerShell:

```powershell
ssh -i "C:\Users\pg9\Desktop\EproVoiceAgentDemo-key.pem" ubuntu@3.26.230.195
```

Then run:

```bash
cloudflared tunnel --url http://127.0.0.1:7862
```

You will get a URL like:

```text
https://xxxxx.trycloudflare.com
```

Copy that HTTPS link. Keep this terminal open as well.

### 3. Mobile phone test

1. Open the `https://xxxxx.trycloudflare.com` link in Chrome on your phone
2. Tap **Start Handset**
3. Allow microphone permission
4. Start the voice conversation test

### 4. Common issues (English)

| Symptom | Cause | Fix |
|---|---|---|
| 502 Bad gateway | Voice service is down, or tunnel points to the wrong port | Make sure `python scripts/run_web.py` is running, then check with `curl -I http://127.0.0.1:7862` |
| Public `http://IP:7862` opens, but mic does not work | Browsers block microphone on non-HTTPS public pages | Use the Cloudflare HTTPS tunnel URL |
| SSH permission denied | Wrong `.pem` path or mismatched key | Confirm the key file is `EproVoiceAgentDemo-key.pem` on the Desktop |
| Demo dies after closing a terminal | Service or tunnel stopped | Restart Terminal A and Terminal B |

### 5. First-time setup notes (English)

For a brand-new EC2 instance:

1. Security group inbound rules: `22` (SSH), `7862` (temporary testing)
2. Upload the project:

```powershell
scp -i "C:\Users\pg9\Desktop\EproVoiceAgentDemo-key.pem" -r "C:\Users\pg9\Desktop\Epro\voice-agent-demo" "C:\Users\pg9\Desktop\Epro\repair-order-agent" ubuntu@3.26.230.195:~/
```

3. Install dependencies on the server:

```bash
sudo apt update
sudo apt install -y python3 python3-pip python3-venv
cd ~/voice-agent-demo
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

4. Set in `.env`:

```env
HOST=0.0.0.0
```

5. Install cloudflared if needed:

```bash
curl -L --output cloudflared.deb https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64.deb
sudo dpkg -i cloudflared.deb
cloudflared --version
```

### 6. Notes for demos (English)

- EC2 can host this real-time voice agent.
- For customer phone demos, always use an HTTPS URL.
- `trycloudflare.com` quick tunnels are good for temporary demos.
- For production, use a fixed domain with proper HTTPS.

---

## Quick Checklist / 快速檢查

1. Terminal A: `python scripts/run_web.py` is running
2. Terminal B: `cloudflared tunnel --url http://127.0.0.1:7862` is running
3. Open the printed `https://...trycloudflare.com` link on a phone
4. Allow microphone and test

If the EC2 public IP changes after stop/start, update the SSH/`scp` IP in this file.
