# app/main.py
import os
from flask import Flask, send_from_directory, jsonify

def create_app():
    server = Flask(__name__, static_folder="static", static_url_path="/static")

    @server.route("/")
    def index():
        return server.send_static_file("index.html")

    @server.route("/assets/<path:filename>")
    def assets(filename):
        return send_from_directory(os.path.join(server.static_folder, "assets"), filename)

    # mount dash apps
    # lazy-import Dash apps to avoid heavy imports at module import time
    from app.dash_apps.farfield import create_farfield_dash
    from app.dash_apps.farfield_range_beams import create_farfield_range_beams_dash
    from app.dash_apps.nearfield_range_beams import create_nearfield_range_beams_dash
    from app.dash_apps.nearfield import create_nearfield_dash
    from app.dash_apps.farfield_range_of_beams_sub6 import create_farfield_range_of_beams_sub6_dash

    create_farfield_dash(server, url_base_pathname="/dash/far_field/")
    create_farfield_range_beams_dash(server, url_base_pathname="/dash/far_field_range_of_beams/")
    create_nearfield_dash(server, url_base_pathname="/dash/near_field/")
    create_nearfield_range_beams_dash(server, url_base_pathname="/dash/near_field_range_of_beams/")
    create_farfield_range_of_beams_sub6_dash(server, url_base_pathname="/dash/far_field_range_of_beams_sub6/")

    @server.route("/api/health")
    def health():
        return jsonify({"status": "ok"})

    return server

app = create_app()

if __name__ == "__main__":
    debug_flag = os.environ.get("DEBUG", os.environ.get("FLASK_DEBUG", "False")).lower() in ("1", "true", "yes")
    port = int(os.environ.get("PORT", 8050))
    app.run(host="0.0.0.0", port=port, debug=debug_flag)
