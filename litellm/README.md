# LiteLLM setup
## general setup
copy .env.example to .env
Change User, Password, Keys

## setup users and keys from shell
- docker-compose.yml: uncomment ports
- start container

User setup:
```bash
curl --location 'http://localhost:4000/user/new' \
  --header 'Authorization: Bearer <YOUR_MASTER_KEY>' \
  --header 'Content-Type: application/json' \
  --data-raw '{
    "user_id": "openwebui"
  }'
```

Virtual Key setup:
```bash
curl --location 'http://localhost:4000/key/generate' \
  --header 'Authorization: Bearer <YOUR_MASTER_KEY>' \
  --header 'Content-Type: application/json' \
  --data-raw '{
    "user_id": "openwebui"
  }'
```
find "key":"sk-SOMETHING" - use this key in OpenWebUI. 

# OWUI Setup
URL: http://litellm-proxy:4000/v1
Auth: Bearer: your virtual key

see https://docs.litellm.ai/docs/tutorials/openweb_ui
