# REST API 設計

## 動画関連

### GET /api/videos/{video_id}
**認証**: 不要  
**説明**: 指定IDの動画メタ情報を取得する。

リクエスト例:
```
curl -X GET https://example.com/api/videos/123
```

レスポンス例:
```json
{
  "id": 123,
  "filename": "swing1.mp4",
  "uploaded_at": "2024-08-20T12:34:56Z"
}
```

**エラー**: `404 VIDEO_NOT_FOUND`

### POST /api/videos
**認証**: 不要  
**説明**: スイング動画をアップロードする。

リクエスト例:
```
curl -X POST https://example.com/api/videos \
  -F "file=@/path/to/swing.mp4" \
  -F "type=reference"
```

レスポンス例:
```json
{
  "id": 124,
  "filename": "swing.mp4"
}
```

**エラー**: `400 INVALID_VIDEO_FORMAT`

---

## 解析関連

### POST /api/analyses
**認証**: 不要  
**説明**: 参照動画と比較動画を解析してスコアを算出する。

リクエスト例:
```json
{
  "reference_video_id": 123,
  "compare_video_id": 124
}
```

レスポンス例:
```json
{
  "analysis_id": "a1b2c3",
  "score": {
    "total": 0.87,
    "by_part": {"head": 0.95, "hip": 0.78}
  }
}
```

**エラー**: `422 ANALYSIS_FAILED`

### GET /api/analyses/{analysis_id}
**認証**: 不要  
**説明**: 指定解析IDの結果を取得する。

リクエスト例:
```
curl -X GET https://example.com/api/analyses/a1b2c3
```

レスポンス例:
```json
{
  "analysis_id": "a1b2c3",
  "status": "completed",
  "score": {
    "total": 0.87,
    "by_part": {"head": 0.95, "hip": 0.78}
  }
}
```

**エラー**: `404 ANALYSIS_NOT_FOUND`

---

## メッセージ関連

### GET /api/messages?analysis_id={analysis_id}
**認証**: 不要  
**説明**: 指定解析に紐づくチャット履歴を取得する。

リクエスト例:
```
curl -X GET "https://example.com/api/messages?analysis_id=a1b2c3"
```

レスポンス例:
```json
{
  "analysis_id": "a1b2c3",
  "messages": [
    {"role": "user", "content": "どこを改善すべき？"},
    {"role": "assistant", "content": "トップで右肘が伸びています"}
  ]
}
```

**エラー**: `404 ANALYSIS_NOT_FOUND`

### POST /api/messages
**認証**: 不要  
**説明**: チャットボットへメッセージを送信する。

リクエスト例:
```json
{
  "analysis_id": "a1b2c3",
  "content": "次の練習メニューは？"
}
```

レスポンス例:
```json
{
  "role": "assistant",
  "content": "インパクト時の体重移動を意識しましょう"
}
```

**エラー**: `400 INVALID_REQUEST`

---

## システム使用率関連

### GET /api/system-usage
**認証**: 不要  
**説明**: CPU/GPU/NPUおよびメモリ使用率を取得する。

リクエスト例:
```
curl -X GET https://example.com/api/system-usage
```

レスポンス例:
```json
{
  "cpu": 32.1,
  "gpu": 41.5,
  "npu": 0.0,
  "memory": 58.3
}
```

**エラー**: `500 INTERNAL_ERROR`

---

## エラーレスポンス共通形式
```json
{
  "error": {
    "code": "ERROR_CODE",
    "message": "人間が読める説明"
  }
}
```

