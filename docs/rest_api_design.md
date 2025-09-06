# REST API

すべてのエンドポイントは認証不要で、`data/` ディレクトリ配下のファイルを操作します。

## 動画管理

### `GET /list_videos`
利用可能な `.mp4` ファイルの一覧を JSON 配列で返します。

```
[
  "reference.mp4",
  "current.mp4"
]
```

### `GET /videos/{filename}`
データディレクトリから動画ファイルをストリーミングします。

### `POST /upload_videos`
1 本または 2 本の動画を multipart でアップロードします。

フィールド:
- `reference`: 参照スイング動画
- `current`: 比較スイング動画

レスポンス例:

```json
{
  "reference_file": "reference.mp4",
  "current_file": "current.mp4"
}
```

### `POST /set_videos`
既存の動画ファイルとオプション設定を選択します。リクエスト例:

```json
{
  "reference_file": "reference.mp4",
  "current_file": "current.mp4",
  "device": "CPU",
  "pose_model": "openvino",
  "backend": "openvino"
}
```

レスポンスには選択された device・backend・pose model が含まれます。

## 解析

### `POST /analyze`
選択された動画の姿勢抽出とスコアリングを実行します。レスポンス例:

```json
{
  "score": {
    "total": 0.87,
    "by_part": {"head": 0.95, "hip": 0.78}
  },
  "analysis_complete": true
}
```

### `POST /extractor`
デバッグ用エンドポイント。アップロードまたは既存ファイルからキーポイントを抽出し、フレーム数やサンプルキーポイントを返します。

### `POST /init_chatbot`
解析成果物が揃った後にスイング用チャットボットを初期化します。成功時は `{"status": "ok"}` を返します。

### `GET /chatbot_status`
チャットボットが有効・初期化済み・利用可能かを報告します。

## チャット

### `GET/POST /messages`
スイング特化チャットボットのエンドポイント。
- `POST` ボディ: `{"message": "text"}`
- レスポンス: `{"reply": "assistant reply"}`
- `GET` は会話履歴または初期化メッセージを返します。

### `GET/POST /chat_messages`
`/chat` で利用できる汎用チャットボット。

### `GET/POST /chat_settings`
汎用チャットボットのバックエンドとデバイスを取得・変更します。`POST` 例:

```json
{"backend": "openvino", "device": "CPU"}
```

## システム情報

### `GET /system_usage`
CPU・GPU・NPU・メモリ使用率を返します。

```json
{
  "cpu": 32.1,
  "gpu": 41.5,
  "npu": 0.0,
  "memory": 58.3
}
```

### `GET /debug/cuda`
Torch と CUDA の可視性を報告します。
