from flask import Flask, request, jsonify, render_template, send_from_directory

app = Flask(__name__)
messages = []  # 簡易的にメッセージをメモリに保持


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/messages", methods=["GET", "POST"])
def message_handler():
    if request.method == "POST":
        data = request.get_json()
        messages.append(data["message"])
        return "", 204  # 成功時は空でOK
    else:
        return jsonify(messages)


@app.route("/videos/<path:filename>")
def serve_video(filename):
    """Serve video files from the data directory."""
    return send_from_directory("data", filename)


if __name__ == "__main__":
    app.run(debug=True)
