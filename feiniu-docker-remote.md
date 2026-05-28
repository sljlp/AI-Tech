# 飞牛 Docker 公网远程访问配置

## 概述

将飞牛（<internal-nas-ip>）的 Docker 套接字通过 FRP 隧道暴露到公网，实现远程管理。

## 架构

```
客户端 (Mac/PC)  -->  公网 FRP 服务器 (<public-server-ip>:4998)
                            |
                     FRP 隧道 (frpc)
                            |
                     飞牛本地 TCP 代理 (127.0.0.1:12375)
                            |
                    Docker Unix Socket (/var/run/docker.sock)
```

## 前置条件

- 飞牛已安装 frpc（v0.61.2，位于 `/tmp/frp_0.61.2_linux_amd64/`）
- FRP 服务器信息：
  - 地址：`<public-server-ip>`
  - 端口：`4900`
  - Token：`<your-frp-token>`
- 公网可用端口范围：`4001-4999`

## 配置步骤

### 1. Unix Socket → TCP 转发

由于 frpc 不支持直接转发 Unix Socket，需要用代理将 Docker 的 Unix Socket 转为 TCP 端口。

在飞牛上创建并运行 Python 代理脚本 `/tmp/docker-proxy.py`：

```python
import os, sys, socket, threading

SOCKET_PATH = "/var/run/docker.sock"
LISTEN_PORT = 12375

def handle_client(conn):
    try:
        us = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        us.connect(SOCKET_PATH)
        def forward(src, dst):
            try:
                while True:
                    data = src.recv(4096)
                    if not data:
                        break
                    dst.sendall(data)
            except:
                pass
            finally:
                try: dst.close()
                except: pass
        t1 = threading.Thread(target=forward, args=(conn, us), daemon=True)
        t2 = threading.Thread(target=forward, args=(us, conn), daemon=True)
        t1.start()
        t2.start()
        t1.join()
        t2.join()
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
    finally:
        try: conn.close()
        except: pass

srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
srv.bind(("127.0.0.1", LISTEN_PORT))
srv.listen(5)
print(f"Proxying {SOCKET_PATH} -> 127.0.0.1:{LISTEN_PORT}", flush=True)
while True:
    conn, addr = srv.accept()
    threading.Thread(target=handle_client, args=(conn,), daemon=True).start()
```

启动代理：

```bash
nohup python3 /tmp/docker-proxy.py > /tmp/docker-proxy.log 2>&1 &
```

验证代理是否在监听：

```bash
ss -tlnp | grep 12375
```

### 2. 启动 FRP 隧道

使用 frpc tcp 子命令将本地 12375 端口转发到公网：

```bash
nohup /tmp/frp_0.61.2_linux_amd64/frpc tcp \
  -s <public-server-ip> \
  -P 4900 \
  -t <your-frp-token> \
  -l 12375 \
  -r 4998 \
  -n feiniu-docker \
  > /tmp/frpc.log 2>&1 &
```

验证隧道连接：

```bash
cat /tmp/frpc.log
# 应看到: login to server success, start proxy success
```

### 3. 客户端使用

在任意机器上设置 Docker 环境变量：

```bash
export DOCKER_HOST=tcp://<public-server-ip>:4998
```

或单次使用：

```bash
DOCKER_HOST=tcp://<public-server-ip>:4998 docker ps
```

### 4. 关闭隧道

```bash
# 在飞牛上执行
kill $(pgrep -f 'frpc.*feiniu-docker')
kill $(pgrep -f docker-proxy.py)
```

## 注意事项

### 安全警告

- Docker 套接字拥有宿主机 root 权限，暴露到公网极度危险
- 任何知道 `<public-server-ip>:4998` 的人都可以完全控制飞牛
- 建议仅在需要时开启隧道，使用完毕后立即关闭
- 不要将端口和 token 提交到公开代码仓库

### 限制

- frpc v0.61.2 的 CLI 子命令不支持直接转发 Unix Socket，需要额外的 TCP 代理
- 公网端口只能从 4001-4999 范围内选取
- 不要修改现有 frpc 配置文件（`/home/user/frpc.toml` 等）

### 替代方案

如仅在局域网使用，推荐 SSH 隧道（更安全）：

```bash
# 在客户端执行
ssh -L 2375:/var/run/docker.sock user@<internal-nas-ip>
export DOCKER_HOST=tcp://localhost:2375
```
