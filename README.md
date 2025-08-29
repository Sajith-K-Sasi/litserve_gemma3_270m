## Install Dependencies
`uv sync`

## Test Text Generation
`uv run text_generation.py`

## Test Text Streaming
`uv run text_streaming.py`


## Run Server
### Deploy for free with autoscaling, monitoring, etc...
`uv run lightning deploy server.py --cloud`

### Or run locally (self host anywhere)
```
uv run lightning deploy server.py
# uv run server.py
```

## Run Client
`uv run client.py`
