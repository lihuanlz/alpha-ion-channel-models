# -*- coding: utf-8 -*-
# bat 冒烟：启动 25 秒，抓输出，然后整树掐死（不留后台进程）
import subprocess, time, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
logp = os.path.join(HERE, "_bat_smoke.log")

with open(logp, "w", encoding="gbk", errors="replace") as log:
    p = subprocess.Popen('cmd /c "CiPA全量下载_双击运行.bat"',
                         cwd=HERE, shell=False, stdin=subprocess.DEVNULL,
                         stdout=log, stderr=subprocess.STDOUT)
    time.sleep(25)
    # 先杀 cmd 壳，再按命令行匹配杀 python 子进程（整树）
    subprocess.run(["powershell", "-c",
                    "Get-CimInstance Win32_Process | Where-Object {$_.CommandLine -match 'CiPA'} "
                    "| ForEach-Object {Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue}"],
                   capture_output=True)
    try:
        p.kill()
    except Exception:
        pass

print(open(logp, encoding="gbk", errors="replace").read()[:3000])
# 确认没有残留
r = subprocess.run(["powershell", "-c",
                    "(Get-CimInstance Win32_Process | Where-Object {$_.CommandLine -match 'CiPA全量下载器'}).Count"],
                   capture_output=True, text=True)
print("残留下载进程数:", r.stdout.strip())
os.remove(logp)
