#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""温度看门狗(通用版):监控CPU/GPU温度,超阈值记录嫌疑进程并发邮件警告(只警告不杀).
适用于组内任意Linux设备(100/200/118/153等).用法见同目录 Temp_WatchDog_Usage.txt
日志:与脚本同目录 temp_watchdog.log
"""
import subprocess, os, time, socket

# ===== CONFIG(各设备部署时按需修改) =====
WARN_C      = 85          # 观察期(0819-0820):抬高以纯记录曲线,定阈值后改回
DANGER_C    = 75          # 危险阈值(°C):CPU高温线80°C/GPU降频线95°C前最后告警
COOLDOWN_S  = 30 * 60     # 告警冷却(秒),期间不重发
SMTP_HOST   = "smtp.qq.com"           # SMTP服务器
SMTP_PORT   = 587
MAIL_FROM   = "zhxbayernfan3@qq.com"   # 发件(小号)
MAIL_TO     = "zhxbayernfan3@qq.com"   # 收件(小号发小号,收件箱必留痕)
MAIL_PWD    = "eglsorkdmnxgbihb"        # SMTP授权码
HERE        = os.path.dirname(os.path.abspath(__file__))
LOG_FILE    = os.path.join(HERE, "temp_watchdog.log")      # 告警历史(长期保留,只追加)
OK_FILE     = os.path.join(HERE, "temp_watchdog_ok.log")   # 常规OK巡检(单行覆盖)
NODE_HOURS  = (0, 9, 18)  # 每天这三个整点把OK快照抄送进告警历史存档
# ==================

def log(msg):
    """告警/异常历史:只追加,长期保留."""
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
    with open(LOG_FILE, "a") as f: f.write(line + "\n")
    print(line)

def log_ok(msg):
    """常规巡检记录.(0819-0820观察期:改为追加记录全天温度曲线,用于画baseline图;观察期结束后恢复单行覆盖+节点存档)"""
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} OK {msg}"
    with open(OK_FILE, "a") as f: f.write(line + "\n")   # 观察期:追加
    print(line)

def get_cpu_temp():
    try:  # 优先 lm-sensors 的 Package 温度
        out = subprocess.run(["sensors"], capture_output=True, text=True, timeout=10).stdout
        best = None
        for ln in out.splitlines():
            if "Package id" in ln and "+" in ln:
                best = float(ln.split("+")[1].split("°C")[0]); break
        if best is not None: return best, "sensors(Package)"
    except Exception: pass
    try:  # 降级:thermal_zone x86_pkg_temp
        for z in os.listdir("/sys/class/thermal"):
            p = f"/sys/class/thermal/{z}/type"
            if os.path.exists(p) and open(p).read().strip() == "x86_pkg_temp":
                v = int(open(f"/sys/class/thermal/{z}/temp").read()) / 1000.0
                return v, "thermal_zone(x86_pkg_temp)"
    except Exception: pass
    return None, "N/A"

def get_gpu_temp():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=10).stdout.strip()
        return float(out.splitlines()[0]), "nvidia-smi"
    except Exception:
        return None, "N/A"

def top_cpu_procs(n=3):
    try:
        out = subprocess.run(["ps", "aux", "--sort=-%cpu"], capture_output=True, text=True, timeout=10).stdout
        rows = [l.split(None, 10) for l in out.splitlines()[1:n+1]]
        return [f"{r[10][:60]} (cpu={r[2]}% pid={r[1]})" for r in rows if len(r) > 10]
    except Exception: return []

def top_gpu_procs(n=3):
    try:
        out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=10).stdout.strip()
        if not out: return []
        lines = out.splitlines()[:n]
        res = []
        for ln in lines:
            pid, mem = [x.strip() for x in ln.split(",")]
            try:
                cmd = subprocess.run(["ps", "-p", pid, "-o", "args="], capture_output=True, text=True).stdout.strip()
            except Exception: cmd = "?"
            res.append(f"{cmd[:60]} (gpu_mem={mem} pid={pid})")
        return res
    except Exception: return []

def in_cooldown():
    # 同时看历史日志(发信成功)——冷却只由真实发信驱动
    if not os.path.exists(LOG_FILE): return False
    try:
        for ln in reversed(open(LOG_FILE).read().splitlines()):
            if "ALERT-SENT" in ln:
                t = time.mktime(time.strptime(ln[:19], "%Y-%m-%d %H:%M:%S"))
                return (time.time() - t) < COOLDOWN_S
    except Exception: pass
    return False

def send_mail(subject, body):
    import smtplib
    from email.mime.text import MIMEText
    if "@" not in MAIL_FROM or "@" not in MAIL_TO or MAIL_PWD == "SMTP_AUTH_CODE":
        raise RuntimeError("CONFIG未填:请在脚本头部填好MAIL_FROM/MAIL_TO/MAIL_PWD")
    msg = MIMEText(body, "plain", "utf-8")
    msg["Subject"], msg["From"], msg["To"] = subject, MAIL_FROM, MAIL_TO
    msg["Cc"] = MAIL_FROM   # 抄送发件人自己:QQ邮箱对Cc自己的信会存入"已发送",保证发件箱留痕
    s = smtplib.SMTP(SMTP_HOST, SMTP_PORT); s.starttls()
    s.login(MAIL_FROM, MAIL_PWD); s.send_message(msg); s.quit()

def main():
    cpu_t, cpu_src = get_cpu_temp()
    gpu_t, gpu_src = get_gpu_temp()
    status = f"cpu={cpu_t if cpu_t is not None else 'N/A'}({cpu_src}) gpu={gpu_t if gpu_t is not None else 'N/A'}({gpu_src})"
    peak = max(v for v in (cpu_t, gpu_t) if v is not None) if (cpu_t is not None or gpu_t is not None) else None
    if peak is None:
        log("ERROR: 读不到任何温度值"); return
    if peak < WARN_C:
        log_ok(status)   # 常规巡检:覆盖式单行;0/9/18节点自动存档
        return
    level = "DANGER" if peak >= DANGER_C else "WARN"
    if in_cooldown():
        return  # 冷却期内不再写日志/发信(防刷屏),温度回落后自然恢复
    log(f"{level} {status} warn>={WARN_C}C danger>={DANGER_C}C")
    body = (f"{socket.gethostname()} 温度{'危险' if level=='DANGER' else '预警'}\n{status} (预警线{WARN_C}°C/危险线{DANGER_C}°C)\n\n"
            "CPU占用TOP3:\n" + "\n".join("  " + x for x in top_cpu_procs()) + "\n\n"
            "GPU占用TOP3:\n" + "\n".join("  " + x for x in top_gpu_procs()) + "\n\n"
            "本看门狗只警告不杀进程,请人工处置.")
    try:
        send_mail(f"[温度{level}]{socket.gethostname()} cpu={cpu_t if cpu_t is not None else 'N/A'} gpu={gpu_t if gpu_t is not None else 'N/A'}", body)
        log("ALERT-SENT 邮件已发")
    except Exception as e:
        log(f"ALERT-FAIL 邮件发送失败: {e}")

if __name__ == "__main__":
    main()
