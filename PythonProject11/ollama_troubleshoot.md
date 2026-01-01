# Ollama 问题排查指南

## 问题：ollama list 命令卡住

### 可能原因

1. **Ollama 服务未运行或卡住**
2. **网络连接问题**
3. **配置文件损坏**
4. **端口被占用**

### 解决方案

#### 方案1: 检查 Ollama 服务状态

在 Windows 上，Ollama 通常作为后台服务运行。检查服务状态：

```powershell
# 检查 Ollama 服务
Get-Service | Where-Object {$_.Name -like "*ollama*"}

# 或者查看进程
Get-Process | Where-Object {$_.ProcessName -like "*ollama*"}
```

#### 方案2: 重启 Ollama 服务

```powershell
# 停止 Ollama 服务（如果存在）
Stop-Service ollama -ErrorAction SilentlyContinue

# 或者终止进程
Get-Process | Where-Object {$_.ProcessName -like "*ollama*"} | Stop-Process -Force

# 重新启动 Ollama（通常会自动启动）
# 如果安装了 Ollama，可以手动启动：
# 在开始菜单搜索 "Ollama" 并启动
```

#### 方案3: 检查端口占用

Ollama 默认使用 11434 端口：

```powershell
# 检查端口占用
netstat -ano | findstr :11434

# 如果端口被占用，可以终止占用进程
```

#### 方案4: 重置 Ollama

```powershell
# 1. 完全停止 Ollama
Get-Process | Where-Object {$_.ProcessName -like "*ollama*"} | Stop-Process -Force

# 2. 清除配置文件（谨慎操作）
# Ollama 配置文件通常位于：
# C:\Users\<用户名>\.ollama

# 3. 重新安装或重启 Ollama
```

#### 方案5: 使用超时命令测试

```powershell
# 设置超时时间测试连接
$job = Start-Job -ScriptBlock { ollama list }
Wait-Job $job -Timeout 5
if ($job.State -eq "Running") {
    Stop-Job $job
    Write-Host "命令超时，Ollama 服务可能无响应"
} else {
    Receive-Job $job
}
```

#### 方案6: 检查防火墙/代理设置

如果使用了代理或防火墙，可能会阻止 Ollama 连接：

```powershell
# 检查环境变量
$env:HTTP_PROXY
$env:HTTPS_PROXY
$env:NO_PROXY
```

#### 方案7: 手动启动 Ollama 服务

```powershell
# 如果 Ollama 已安装但服务未启动
# 通常需要从开始菜单或应用程序文件夹启动 Ollama
# 或者使用完整路径：
# & "C:\Users\<用户名>\AppData\Local\Programs\Ollama\ollama.exe" serve
```

### 快速诊断脚本

创建一个 PowerShell 脚本来诊断问题：

```powershell
Write-Host "=== Ollama 诊断 ===" -ForegroundColor Cyan

# 1. 检查进程
Write-Host "`n1. 检查 Ollama 进程:" -ForegroundColor Yellow
$processes = Get-Process | Where-Object {$_.ProcessName -like "*ollama*"}
if ($processes) {
    $processes | Format-Table ProcessName, Id, CPU
} else {
    Write-Host "未找到 Ollama 进程" -ForegroundColor Red
}

# 2. 检查端口
Write-Host "`n2. 检查端口 11434:" -ForegroundColor Yellow
$port = netstat -ano | findstr :11434
if ($port) {
    Write-Host $port
} else {
    Write-Host "端口 11434 未被占用" -ForegroundColor Red
}

# 3. 测试连接
Write-Host "`n3. 测试连接:" -ForegroundColor Yellow
try {
    $response = Invoke-WebRequest -Uri "http://localhost:11434/api/tags" -TimeoutSec 5 -ErrorAction Stop
    Write-Host "连接成功！" -ForegroundColor Green
    Write-Host $response.Content
} catch {
    Write-Host "连接失败: $_" -ForegroundColor Red
}
```

### 最常用的解决方法

**如果 ollama list 卡住，通常按以下顺序尝试：**

1. **强制终止并重启 Ollama**
   ```powershell
   Get-Process | Where-Object {$_.ProcessName -like "*ollama*"} | Stop-Process -Force
   # 然后重新打开 Ollama 应用
   ```

2. **使用 API 测试**
   ```powershell
   # 测试 API 是否响应
   Invoke-WebRequest -Uri "http://localhost:11434/api/tags"
   ```

3. **检查日志**
   - Ollama 日志通常在：`C:\Users\<用户名>\.ollama\logs\`

4. **重新安装 Ollama**
   - 如果以上方法都无效，考虑重新安装 Ollama




