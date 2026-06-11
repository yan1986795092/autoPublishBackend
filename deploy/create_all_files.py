#!/usr/bin/env python3
"""
AutoPublish 部署配置生成脚本
在服务器上运行：python3 create_all_files.py
"""
import base64, os, sys

def write_file(path, content):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if isinstance(content, bytes):
        with open(path, 'wb') as f:
            f.write(content)
    else:
        with open(path, 'w') as f:
            f.write(content)
    print(f"  ✓ {path} ({len(content)} bytes)")

base = '/opt/autopublish'

print("=" * 50)
print("AutoPublish 部署配置生成中...")
print("=" * 50)

# 1. docker-compose.yml (base64 encoded to avoid terminal issues)
docker_compose_b64 = """
dmVyc2lvbjogJzMuOCcKCiMgPT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09
PT09PT09PT09PT09PT09PT09PT09PT09PT09CiMgQXV0b1B1Ymxpc2ggLSDlhajmoIjp
g6jnvbLphY3nva4KIyDln7rkuo4gMuaguDRHIOi1hOa6kOS8mOWMlu+8jOaJgOacieac
jeWKoeaVtOWQiOWcqOWQjOS4gOWPsCBMQVMg5a6e5L6LCiMgPT09PT09PT09PT09PT09
PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09CgpuZXR3
b3JrczoKICBhdXRvcHVibGlzaC1uZXQ6CiAgICBkcml2ZXI6IGJyaWRnZQogICAgZHJp
dmVyX29wdHM6CiAgICAgIGNvbS5kb2NrZXIubmV0d29yay5kcml2ZXIubXR1OiAxNDgw
Cgp2b2x1bWVzOgogIHBvc3RncmVzLWRhdGE6CiAgcmVkaXMtZGF0YToKICBlcy1kYXRh
OgogIHByb21ldGhldXMtZGF0YToKICBncmFmYW5hLWRhdGE6CiAgYmFja2VuZC1kYXRh
OgoKc2VydmljZXM6CgogICMg4pSA4pSA4pSAIFBvc3RncmVTUUzvvJrkuLvmlbDmja7l
upMg4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA
4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSACiAgcG9zdGdy
ZXM6CiAgICBpbWFnZTogcG9zdGdyZXM6MTYtYWxwaW5lCiAgICBjb250YWluZXJfbmFt
ZTogYXV0b3B1Ymxpc2gtcG9zdGdyZXMKICAgIHJlc3RhcnQ6IHVubGVzcy1zdG9wcGVk
CiAgICBuZXR3b3JrczoKICAgICAgLSBhdXRvcHVibGlzaC1uZXQKICAgIHBvcnRzOgog
ICAgICAtICI1NDMyOjU0MzIiCiAgICBlbnZpcm9ubWVudDoKICAgICAgUE9TVEdSRVNf
REI6IGF1dG9wdWJsaXNoCiAgICAgIFBPU1RHUkVTX1VTRVI6IGF1dG9wdWJsaXNoCiAg
ICAgIFBPU1RHUkVTX1BBU1NXT1JEOiAke0RCX1BBU1NXT1JEOi1hdXRvcHVibGlzaF9w
cm9kXzIwMjR9CiAgICB2b2x1bWVzOgogICAgICAtIHBvc3RncmVzLWRhdGE6L3Zhci9s
aWIvcG9zdGdyZXNxbC9kYXRhCiAgICBoZWFsdGhjaGVjazoKICAgICAgdGVzdDogWyJD
TUQtU0hFTEwiLCAicGdfaXNyZWFkeSAtVSBhdXRvcHVibGlzaCAtZCBhdXRvcHVibGlz
aCJdCiAgICAgIGludGVydmFsOiAxMHMKICAgICAgdGltZW91dDogNXMKICAgICAgcmV0
cmllczogNQogICAgZGVwbG95OgogICAgICByZXNvdXJjZXM6CiAgICAgICAgbGltaXRz
OgogICAgICAgICAgbWVtb3J5OiA1MTJNCiAgICAgICAgICBjcHVzOiAnMC41JwoKICAj
IOKUgOKUgOKUgCBSZWRpc++8mue8k+WtmCAvIOS7u+WKoemYn+WIlyDilIDilIDilIDi
lIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDi
lIDilIDilIDilIDilIDilIAKICByZWRpczoKICAgIGltYWdlOiByZWRpczo3LWFscGlu
ZQogICAgY29udGFpbmVyX25hbWU6IGF1dG9wdWJsaXNoLXJlZGlzCiAgICByZXN0YXJ0
OiB1bmxlc3Mtc3RvcHBlZAogICAgbmV0d29ya3M6CiAgICAgIC0gYXV0b3B1Ymxpc2gt
bmV0CiAgICBwb3J0czoKICAgICAgLSAiNjM3OTo2Mzc5IgogICAgY29tbWFuZDogcmVk
aXMtc2VydmVyIC0tc2F2ZSAiIiAtLWFwcGVuZG9ubHkgbm8gLS1tYXhtZW1vcnkgNjRt
YiAtLW1heG1lbW9yeS1wb2xpY3kgYWxsa2V5cy1scnUKICAgIHZvbHVtZXM6CiAgICAg
IC0gcmVkaXMtZGF0YTovZGF0YQogICAgaGVhbHRoY2hlY2s6CiAgICAgIHRlc3Q6IFsi
Q01EIiwgInJlZGlzLWNsaSIsICJwaW5nIl0KICAgICAgaW50ZXJ2YWw6IDEwcwogICAg
ICB0aW1lb3V0OiAzcwogICAgICByZXRyaWVzOiA1CiAgICBkZXBsb3k6CiAgICAgIHJl
c291cmNlczoKICAgICAgICBsaW1pdHM6CiAgICAgICAgICBtZW1vcnk6IDEyOE0KICAg
ICAgICAgIGNwdXM6ICcwLjI1JwoKICAjIOKUgOKUgOKUgCBFbGFzdGljc2VhcmNo77ya
5pel5b+XIC8gVHJhY2Ug5a2Y5YKoIOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKU
gOKUgOKUgOKUgOKUgOKUgOKUgOKUgAogIGVsYXN0aWNzZWFyY2g6CiAgICBpbWFnZTog
ZWxhc3RpY3NlYXJjaDo4LjEzLjAKICAgIGNvbnRhaW5lcl9uYW1lOiBhdXRvcHVibGlz
aC1lcwogICAgcmVzdGFydDogdW5sZXNzLXN0b3BwZWQKICAgIG5ldHdvcmtzOgogICAg
ICAtIGF1dG9wdWJsaXNoLW5ldAogICAgcG9ydHM6CiAgICAgIC0gIjkyMDA6OTIwMCIK
ICAgIGVudmlyb25tZW50OgogICAgICAtIGRpc2NvdmVyeS50eXBlPXNpbmdsZS1ub2Rl
CiAgICAgIC0geHBhY2suc2VjdXJpdHkuZW5hYmxlZD1mYWxzZQogICAgICAtICJFU19K
QVZBX09QVFM9LVhtczI1Nm0gLVhteDI1Nm0iCiAgICAgIC0gY2x1c3Rlci5uYW1lPWF1
dG9wdWJsaXNoLW9icwogICAgICAtIGFjdGlvbi5hdXRvX2NyZWF0ZV9pbmRleD10cnVl
CiAgICAgIC0gaW5kaWNlcy5tZW1vcnkuaW5kZXhfYnVmZmVyX3NpemU9MTAlCiAgICAg
IC0gaW5kaWNlcy5maWVsZGRhdGEuY2FjaGUuc2l6ZT0yMCUKICAgIHZvbHVtZXM6CiAg
ICAgIC0gZXMtZGF0YTovdXNyL3NoYXJlL2VsYXN0aWNzZWFyY2gvZGF0YQogICAgdWxp
bWl0czoKICAgICAgbWVtbG9jazoKICAgICAgICBzb2Z0OiAtMQogICAgICAgIGhhcmQ6
IC0xCiAgICAgIG5vZmlsZToKICAgICAgICBzb2Z0OiA2NTUzNgogICAgICAgIGhhcmQ6
IDY1NTM2CiAgICBoZWFsdGhjaGVjazoKICAgICAgdGVzdDogWyJDTUQtU0hFTEwiLCAi
Y3VybCAtcyBodHRwOi8vbG9jYWxob3N0OjkyMDAvX2NsdXN0ZXIvaGVhbHRoIHwgZ3Jl
cCAtcSAnXCJzdGF0dXNcIjpcImdyZWVuXCJcXHxcInN0YXR1c1wiOlwieWVsbG93XCIn
Il0KICAgICAgaW50ZXJ2YWw6IDE1cwogICAgICB0aW1lb3V0OiA1cwogICAgICByZXRy
aWVzOiAxMAogICAgZGVwbG95OgogICAgICByZXNvdXJjZXM6CiAgICAgICAgbGltaXRz
OgogICAgICAgICAgbWVtb3J5OiAxZwogICAgICAgICAgY3B1czogJzAuNScKCiAgIyDi
lIDilIDilIAgTmdpbnjvvJrlj43lkJHku6PnkIYgKyDliY3nq6/pnZnmgIHmlofku7Yg
4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSACiAg
bmdpbng6CiAgICBidWlsZDoKICAgICAgY29udGV4dDogLgogICAgICBkb2NrZXJmaWxl
OiBkZXBsb3kvRG9ja2VyZmlsZS5uZ2lueAogICAgY29udGFpbmVyX25hbWU6IGF1dG9w
dWJsaXNoLW5naW54CiAgICByZXN0YXJ0OiB1bmxlc3Mtc3RvcHBlZAogICAgbmV0d29y
a3M6CiAgICAgIC0gYXV0b3B1Ymxpc2gtbmV0CiAgICBwb3J0czoKICAgICAgLSAiODA4
MDo4MCIKICAgIGRlcGVuZHNfb246CiAgICAgIC0gYmFja2VuZAogICAgICAtIGxsbS1n
YXRld2F5CiAgICBkZXBsb3k6CiAgICAgIHJlc291cmNlczoKICAgICAgICBsaW1pdHM6
CiAgICAgICAgICBtZW1vcnk6IDY0TQogICAgICAgICAgY3B1czogJzAuMjUnCgogICMg
4pSA4pSA4pSAIEJhY2tlbmQgQVBJ77yaQXV0b1B1Ymxpc2gg5qC45b+DIOKUgOKUgOKU
gOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgAogIGJh
Y2tlbmQ6CiAgICBidWlsZDoKICAgICAgY29udGV4dDogLi9iYWNrZW5kCiAgICAgIGRv
Y2tlcmZpbGU6IERvY2tlcmZpbGUKICAgIGNvbnRhaW5lcl9uYW1lOiBhdXRvcHVibGlz
aC1hcGkKICAgIHJlc3RhcnQ6IHVubGVzcy1zdG9wcGVkCiAgICBuZXR3b3JrczoKICAg
ICAgLSBhdXRvcHVibGlzaC1uZXQKICAgIHBvcnRzOgogICAgICAtICI5MDAwOjkwMDAi
CiAgICBlbnZfZmlsZToKICAgICAgLSAuL2JhY2tlbmQvLmVudi5wcm9kdWN0aW9uCiAg
ICBlbnZpcm9ubWVudDoKICAgICAgLSBEQVRBQkFTRV9VUkw9cG9zdGdyZXNxbCtwc3lj
b3BnMjovL2F1dG9wdWJsaXNoOmF1dG9wdWJsaXNoX3Byb2RfMjAyNEBwb3N0Z3Jlczo1
NDMyL2F1dG9wdWJsaXNoCiAgICAgIC0gUkVESVNfVVJMPXJlZGlzOi8vcmVkaXM6NjM3
OS8wCiAgICAgIC0gRUxBU1RJQ1NFQVJDSF9VUkw9aHR0cDovL2VsYXN0aWNzZWFyY2g6
OTIwMAogICAgICAtIE9URUxfU0VSVklDRV9OQU1FPWF1dG9wdWJsaXNoLWFwaQogICAg
ICAtIERFUExPWU1FTlRfRU5WSVJPTk1FTlQ9cHJvZHVjdGlvbgogICAgICAtIERFQlVH
PWZhbHNlCiAgICAgIC0gTE9HX0pTT049dHJ1ZQogICAgdm9sdW1lczoKICAgICAgLSBi
YWNrZW5kLWRhdGE6L2FwcC9kYXRhCiAgICBkZXBlbmRzX29uOgogICAgICBwb3N0Z3Jl
czoKICAgICAgICBjb25kaXRpb246IHNlcnZpY2VfaGVhbHRoeQogICAgICByZWRpczoK
ICAgICAgICBjb25kaXRpb246IHNlcnZpY2VfaGVhbHRoeQogICAgaGVhbHRoY2hlY2s6
CiAgICAgIHRlc3Q6IFsiQ01EIiwgImN1cmwiLCAiLWYiLCAiaHR0cDovL2xvY2FsaG9z
dDo5MDAwL2hlYWx0aCJdCiAgICAgIGludGVydmFsOiAzMHMKICAgICAgdGltZW91dDog
MTBzCiAgICAgIHJldHJpZXM6IDMKICAgIGRlcGxveToKICAgICAgcmVzb3VyY2VzOgog
ICAgICAgIGxpbWl0czoKICAgICAgICAgIG1lbW9yeTogNTEyTQogICAgICAgICAgY3B1
czogJzEuMCcKCiAgIyDilIDilIDilIAgTExNIEdhdGV3YXnvvJpMTE0g6Lev55SxIC8g
6ZmQ5rWBIC8g54aU5patIOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgAogIGxs
bS1nYXRld2F5OgogICAgYnVpbGQ6CiAgICAgIGNvbnRleHQ6IC4vbGxtLWdhdGV3YXkK
ICAgICAgZG9ja2VyZmlsZTogRG9ja2VyZmlsZQogICAgY29udGFpbmVyX25hbWU6IGF1
dG9wdWJsaXNoLWxsbS1nYXRld2F5CiAgICByZXN0YXJ0OiB1bmxlc3Mtc3RvcHBlZAog
ICAgbmV0d29ya3M6CiAgICAgIC0gYXV0b3B1Ymxpc2gtbmV0CiAgICBwb3J0czoKICAg
ICAgLSAiOTAxMDo5MDEwIgogICAgZW52X2ZpbGU6CiAgICAgIC0gLi9sbG0tZ2F0ZXdh
eS8uZW52LnByb2R1Y3Rpb24KICAgIGVudmlyb25tZW50OgogICAgICAtIFJFRElTX1VS
TD1yZWRpczovL3JlZGlzOjYzNzkvMQogICAgICAtIEVMQVNUSUNTSEFSQ0hfVVJMPWh0
dHA6Ly9lbGFzdGljc2VhcmNoOjkyMDAKICAgICAgLSBERVBMT1lNRU5UX0VOVklST05N
RU5UPXByb2R1Y3Rpb24KICAgIGRlcGVuZHNfb246CiAgICAgIHJlZGlzOgogICAgICAg
IGNvbmRpdGlvbjogc2VydmljZV9oZWFsdGh5CiAgICBoZWFsdGhjaGVjazoKICAgICAg
dGVzdDogWyJDTUQiLCAiY3VybCIsICItZiIsICJodHRwOi8vbG9jYWxob3N0OjkwMTAv
aGVhbHRoIl0KICAgICAgaW50ZXJ2YWw6IDMwcwogICAgICB0aW1lb3V0OiAxMHMKICAg
ICAgcmV0cmllczogMwogICAgZGVwbG95OgogICAgICByZXNvdXJjZXM6CiAgICAgICAg
bGltaXRzOgogICAgICAgICAgbWVtb3J5OiAyNTZNCiAgICAgICAgICBjcHVzOiAnMC41
JwoKICAjIOKUgOKUgOKUgCBQcm9tZXRoZXVz77ya5oyH5qCH6YeH6ZuGIOKUgOKUgOKU
gOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKU
gOKUgOKUgOKUgOKUgOKUgOKUgAogIHByb21ldGhldXM6CiAgICBpbWFnZTogcHJvbS9w
cm9tZXRoZXVzOnYyLjU1LjAKICAgIGNvbnRhaW5lcl9uYW1lOiBhdXRvcHVibGlzaC1w
cm9tZXRoZXVzCiAgICByZXN0YXJ0OiB1bmxlc3Mtc3RvcHBlZAogICAgbmV0d29ya3M6
CiAgICAgIC0gYXV0b3B1Ymxpc2gtbmV0CiAgICBwb3J0czoKICAgICAgLSAiOTA5MDo5
MDkwIgogICAgdm9sdW1lczoKICAgICAgLSAuL2RlcGxveS9wcm9tZXRoZXVzLnltbDov
ZXRjL3Byb21ldGhldXMvcHJvbWV0aGV1cy55bWw6cm8KICAgICAgLSAuL2RlcGxveS9p
bmZyYS9wcm9tZXRoZXVzL2FsZXJ0cy55bWw6L2V0Yy9wcm9tZXRoZXVzL2FsZXJ0cy55
bWw6cm8KICAgICAgLSBwcm9tZXRoZXVzLWRhdGE6L3Byb21ldGhldXMKICAgIGNvbW1h
bmQ6CiAgICAgIC0gJy0tY29uZmlnLmZpbGU9L2V0Yy9wcm9tZXRoZXVzL3Byb21ldGhl
dXMueW1sJwogICAgICAtICctLXN0b3JhZ2UudHNkYi5yZXRlbnRpb24udGltZT03ZCcK
ICAgICAgLSAnLS1zdG9yYWdlLnRzZGIucmV0ZW50aW9uLnNpemU9MUdCJwogICAgICAt
ICctLXdlYi5lbmFibGUtbGlmZWN5Y2xlJwogICAgZGVwbG95OgogICAgICByZXNvdXJj
ZXM6CiAgICAgICAgbGltaXRzOgogICAgICAgICAgbWVtb3J5OiAyNTZNCiAgICAgICAg
ICBjcHVzOiAnMC4yNScKCiAgIyDilIDilIDilIAgR3JhZmFuYbvvn5iv6KeG5YyW55yL
5p2/IOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKU
gOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgAogIGdyYWZhbmE6CiAgICBp
bWFnZTogZ3JhZmFuYS9ncmFmYW5hOjExLjUuMgogICAgY29udGFpbmVyX25hbWU6IGF1
dG9wdWJsaXNoLWdyYWZhbmEKICAgIHJlc3RhcnQ6IHVubGVzcy1zdG9wcGVkCiAgICBu
ZXR3b3JrczoKICAgICAgLSBhdXRvcHVibGlzaC1uZXQKICAgIHBvcnRzOgogICAgICAt
ICIzMDAwOjMwMDAiCiAgICBlbnZpcm9ubWVudDoKICAgICAgLSBHRl9TRUNVUklUWV9B
RE1JTl9VU0VSPWFkbWluCiAgICAgIC0gR0ZfU0VDVVJJVFlfQURNSU5fUEFTU1dPUkQ9
YWRtaW4KICAgICAgLSBHRl9VU0VSU19BTExPV19TSUdOX1VQPWZhbHNlCiAgICAgIC0g
R0ZfRkVBVFVSRV9UT0dHTEVTX0VOQUJMRT10cmFjZXFsRWRpdG9yCiAgICB2b2x1bWVz
OgogICAgICAtIC4vZGVwbG95L2luZnJhL2dyYWZhbmEvcHJvdmlzaW9uaW5nOi9ldGMv
Z3JhZmFuYS9wcm92aXNpb25pbmc6cm8KICAgICAgLSBncmFmYW5hLWRhdGE6L3Zhci9s
aWIvZ3JhZmFuYQogICAgZGVwZW5kc19vbjoKICAgICAgLSBwcm9tZXRoZXVzCiAgICAg
IC0gZWxhc3RpY3NlYXJjaAogICAgZGVwbG95OgogICAgICByZXNvdXJjZXM6CiAgICAg
ICAgbGltaXRzOgogICAgICAgICAgbWVtb3J5OiAxMjhNCiAgICAgICAgICBjcHVzOiAn
MC4yNScK"""

write_file(f"{base}/deploy/docker-compose.yml", base64.b64decode(docker_compose_b64))

# 2. Nginx config
nginx_conf = """server {
    listen 80;
    server_name _;
    root /usr/share/nginx/html;
    index index.html;

    gzip on;
    gzip_types text/css application/javascript application/json image/svg+xml;
    gzip_min_length 256;

    location /api/ {
        proxy_pass http://backend:9000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_buffering off;
        proxy_cache off;
        proxy_read_timeout 300s;
    }

    location /gateway/ {
        proxy_pass http://llm-gateway:9010;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }

    location /docs { proxy_pass http://backend/docs; proxy_set_header Host $host; }
    location /redoc { proxy_pass http://backend/redoc; proxy_set_header Host $host; }
    location /openapi.json { proxy_pass http://backend/openapi.json; proxy_set_header Host $host; }
    location /health { proxy_pass http://backend/health; proxy_set_header Host $host; }
    location /metrics { proxy_pass http://backend/metrics; proxy_set_header Host $host; }

    location / {
        try_files $uri $uri/ /index.html;
    }

    add_header X-Frame-Options "SAMEORIGIN";
    add_header X-Content-Type-Options "nosniff";
    add_header X-XSS-Protection "1; mode=block";
}
"""
write_file(f"{base}/deploy/nginx.conf", nginx_conf)

# 3. Dockerfile.nginx
dockerfile_nginx = """FROM node:20-alpine AS builder
WORKDIR /app
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci && npm cache clean --force
COPY frontend/ .
RUN npm run build

FROM nginx:alpine
COPY --from=builder /app/dist /usr/share/nginx/html
COPY deploy/nginx.conf /etc/nginx/conf.d/default.conf
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
"""
write_file(f"{base}/deploy/Dockerfile.nginx", dockerfile_nginx)

# 4. Prometheus config
prometheus_yml = """global:
  scrape_interval: 15s
  evaluation_interval: 15s
rule_files:
  - 'alerts.yml'
scrape_configs:
  - job_name: 'prometheus'
    static_configs:
      - targets: ['localhost:9090']
  - job_name: 'autopublish-api'
    metrics_path: /metrics
    static_configs:
      - targets: ['backend:9000']
    relabel_configs:
      - source_labels: [__address__]
        target_label: instance
        replacement: 'autopublish-api'
  - job_name: 'llm-gateway'
    metrics_path: /metrics
    static_configs:
      - targets: ['llm-gateway:9010']
    relabel_configs:
      - source_labels: [__address__]
        target_label: instance
        replacement: 'llm-gateway'
"""
write_file(f"{base}/deploy/prometheus.yml", prometheus_yml)

# 5. Prometheus alerts
alerts_yaml = """groups:
  - name: autopublish_alerts
    interval: 30s
    rules:
      - alert: HighErrorRate
        expr: (sum(rate(agent_run_total{status="failed"}[5m])) / sum(rate(agent_run_total[5m]))) > 0.05
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "Agent error rate > 5% for 5 minutes"
      - alert: ServiceDown
        expr: absent(up{job="autopublish-api"}) == 1
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "AutoPublish API is down"
"""
write_file(f"{base}/deploy/infra/prometheus/alerts.yml", alerts_yaml)

# 6. Grafana datasources
datasources_yaml = """apiVersion: 1
datasources:
  - name: Prometheus
    type: prometheus
    access: proxy
    url: http://prometheus:9090
    isDefault: true
    editable: false
  - name: Elasticsearch
    type: elasticsearch
    access: proxy
    url: http://elasticsearch:9200
    database: traces-*
    jsonData:
      esVersion: 8.13.0
      timeField: startTimeMillis
    editable: false
"""
write_file(f"{base}/deploy/infra/grafana/provisioning/datasources/datasources.yaml", datasources_yaml)

# 7. Grafana dashboards config
dashboards_yaml = """apiVersion: 1
providers:
  - name: 'default'
    folder: 'Agent Observability'
    type: file
    options:
      path: /etc/grafana/provisioning/dashboards
"""
write_file(f"{base}/deploy/infra/grafana/provisioning/dashboards/dashboards.yaml", dashboards_yaml)

# 8. Backend .env.production
backend_env = """APP_NAME=AutoPublish
DEBUG=false
SECRET_KEY=change-this-to-a-random-secret-key-min-32-chars
LOG_JSON=true
# DB is set by docker-compose env vars
# LLM API Keys - fill in your actual keys
ZHIPU_API_KEY=
ZHIPU_MODEL=glm-4-flash
ZHIPU_BASE_URL=https://open.bigmodel.cn/api/paas/v4
DEFAULT_LLM_PROVIDER=zhipu
BOCHA_API_KEY=
OTEL_SERVICE_NAME=autopublish-api
ELASTICSEARCH_URL=http://elasticsearch:9200
DEPLOYMENT_ENVIRONMENT=production
"""
write_file(f"{base}/backend/.env.production", backend_env)

# 9. LLM Gateway .env.production
gateway_env = """LOG_LEVEL=INFO
LOG_JSON_FORMAT=true
REDIS_URL=redis://redis:6379/1
ELASTICSEARCH_URL=http://elasticsearch:9200
DEPLOYMENT_ENVIRONMENT=production
OTEL_SERVICE_NAME=llm-gateway
"""
write_file(f"{base}/llm-gateway/.env.production", gateway_env)

# 10. Add psycopg2-binary to requirements.txt
req_path = f"{base}/backend/requirements.txt"
with open(req_path) as f:
    req_content = f.read()
if "psycopg2" not in req_content:
    req_content += "\n# PostgreSQL driver\npsycopg2-binary>=2.9.9\n"
    with open(req_path, 'w') as f:
        f.write(req_content)
    print(f"  ✓ {req_path} (added psycopg2-binary)")
else:
    print(f"  ~ {req_path} (psycopg2 already present)")

print()
print("=" * 50)
print("所有配置文件已创建！")
print("现在运行：cd /opt/autopublish && docker-compose -f deploy/docker-compose.yml build")
print("=" * 50)
