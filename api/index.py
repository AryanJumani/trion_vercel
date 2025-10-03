from flask import Flask, request, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_bcrypt import Bcrypt
import cv2
import numpy as np
import base64
from flask_jwt_extended import create_access_token, JWTManager
import os
from waitress import serve

app = Flask(__name__)
app.config['JWT_SECRET_KEY'] = os.environ.get("JWT_KEY", "YOUR_SECRET_KEY")
jwt = JWTManager(app)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)
bcrypt = Bcrypt(app)

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password_hash = db.Column(db.String(128), nullable=False)

    def __init__(self, username, password):
        self.username = username
        # Never store plain text passwords! Always hash them.
        self.password_hash = bcrypt.generate_password_hash(password).decode('utf-8')

    def check_password(self, password):
        return bcrypt.check_password_hash(self.password_hash, password)


with app.app_context():
    db.create_all()




def create_uv_layout(image_bytes):
    # This is the only change to your input method.
    # It decodes the image data sent from Unity instead of reading from a local file.
    nparr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)

    if image is None:
        print("Error: Could not decode image data sent from client.")
        return None

    # ==========================================================
    # YOUR UNCHANGED CODE STARTS HERE
    # ==========================================================

    alpha_mask = image[:, :, 3]
    height, width = alpha_mask.shape
    column_opacity = (
            np.sum(alpha_mask > 0, axis=0) / height
    )  # percentage of opaque pixels per column

    bgr = image[:, :, :3]

    torso_mask = np.zeros_like(alpha_mask, dtype=np.uint8)
    left_sleeve_mask = np.zeros_like(alpha_mask, dtype=np.uint8)
    right_sleeve_mask = np.zeros_like(alpha_mask, dtype=np.uint8)

    torso_threshold = 0.5
    torso_columns = np.where(column_opacity > torso_threshold)[0]

    if len(torso_columns) > 0:
        torso_start = torso_columns[0]
        torso_end = torso_columns[-1]
    else:
        torso_start, torso_end = 0, 0

    torso_mask[:, torso_start: torso_end + 1] = alpha_mask[:, torso_start: torso_end + 1]

    left_sleeve_columns = np.where((column_opacity > 0) & (np.arange(width) < torso_start))[
        0
    ]
    if len(left_sleeve_columns) > 0:
        left_start = left_sleeve_columns[0]
        left_end = left_sleeve_columns[-1]
        left_sleeve_mask[:, left_start: left_end + 1] = alpha_mask[
                                                        :, left_start: left_end + 1
                                                        ]

    right_sleeve_columns = np.where((column_opacity > 0) & (np.arange(width) > torso_end))[
        0
    ]
    if len(right_sleeve_columns) > 0:
        right_start = right_sleeve_columns[0]
        right_end = right_sleeve_columns[-1]
        right_sleeve_mask[:, right_start: right_end + 1] = alpha_mask[
                                                           :, right_start: right_end + 1
                                                           ]

    def crop_mask_and_image(bgr, alpha_mask):
        ys, xs = np.where(alpha_mask > 0)
        if len(xs) == 0 or len(ys) == 0:
            return None, None
        x_min, x_max = xs.min(), xs.max()
        y_min, y_max = ys.min(), ys.max()
        cropped_bgr = bgr[y_min: y_max + 1, x_min: x_max + 1]
        cropped_alpha = alpha_mask[y_min: y_max + 1, x_min: x_max + 1]
        return cropped_bgr, cropped_alpha

    def resize_part(part_bgr, part_alpha, target_width):
        h, w = part_alpha.shape
        part_bgr = cv2.resize(part_bgr, (target_width, h), interpolation=cv2.INTER_LINEAR)
        part_alpha = cv2.resize(
            part_alpha, (target_width, h), interpolation=cv2.INTER_NEAREST
        )
        return part_bgr, part_alpha

    torso_bgr, torso_alpha = crop_mask_and_image(bgr, torso_mask)
    left_sleeve_bgr, left_alpha = crop_mask_and_image(bgr, left_sleeve_mask)
    right_sleeve_bgr, right_alpha = crop_mask_and_image(bgr, right_sleeve_mask)

    torso_bgr, torso_alpha = resize_part(torso_bgr, torso_alpha, 428)
    left_sleeve_bgr, left_alpha = resize_part(left_sleeve_bgr, left_alpha, 366)
    right_sleeve_bgr, right_alpha = resize_part(right_sleeve_bgr, right_alpha, 862)

    canvas = np.zeros((1024, 1024, 4), dtype=np.uint8)

    def place_part(part_bgr, part_alpha, canvas, x, y):
        h, w = part_alpha.shape
        canvas_h, canvas_w = canvas.shape[:2]

        end_x = min(x + w, canvas_w)
        end_y = min(y + h, canvas_h)
        w = end_x - x
        h = end_y - y
        if w <= 0 or h <= 0:
            return
        part_rgba = np.dstack((part_bgr[:h, :w], part_alpha[:h, :w]))

        roi = canvas[y:end_y, x:end_x]

        alpha_part = part_rgba[:, :, 3:4] / 255.0
        alpha_canvas = roi[:, :, 3:4] / 255.0

        out_alpha = alpha_part + alpha_canvas * (1 - alpha_part)
        out_rgb = (
                          part_rgba[:, :, :3] * alpha_part
                          + roi[:, :, :3] * alpha_canvas * (1 - alpha_part)
                  ) / (out_alpha + 1e-6)

        canvas[y:end_y, x:end_x, :3] = out_rgb.astype(np.uint8)
        canvas[y:end_y, x:end_x, 3] = (out_alpha[:, :, 0] * 255).astype(np.uint8)

    place_part(torso_bgr, torso_alpha, canvas, 20, 456)
    place_part(torso_bgr, torso_alpha, canvas, 525, 456)
    place_part(left_sleeve_bgr, left_alpha, canvas, 85, 186)
    place_part(right_sleeve_bgr, right_alpha, canvas, 578, 186)

    # ==========================================================
    # YOUR UNCHANGED CODE ENDS HERE
    # ==========================================================

    # This is the only change to your output method.
    # It encodes the final canvas to a PNG in memory to be sent back to Unity.
    _, buffer = cv2.imencode('.png', canvas)
    return buffer.tobytes()


@app.route('/process_image', methods=['POST'])
def process_image_endpoint():
    """Handles the web request from Unity."""
    if 'image' not in request.json:
        return jsonify({'error': 'No image provided in JSON body'}), 400

    try:
        # Decode the Base64 string sent from Unity
        image_data = base64.b64decode(request.json['image'])

        # Process the image using your exact logic
        processed_image_bytes = create_uv_layout(image_data)

        if processed_image_bytes is None:
            return jsonify({'error': 'Function create_uv_layout returned None. Check server logs.'}), 500

        # Encode the resulting image bytes back to Base64 to send in JSON
        processed_image_base64 = base64.b64encode(processed_image_bytes).decode('utf-8')

        return jsonify({'image': processed_image_base64})

    except Exception as e:
        # This will catch errors inside your processing logic if they occur
        print(f"An error occurred during image processing: {e}")
        return jsonify({'error': 'An internal server error occurred. Check server logs for details.'}), 500

@app.route('/')
def hello():
    return 'Hello World'

@app.route('/login', methods=['POST'])
def login():
    if not request.is_json:
        return jsonify({"msg": "Missing JSON in request"}), 400
    data = request.get_json()
    username = data.get('username')
    password = data.get('password')
    if not username or not password:
        return jsonify({"msg": "Username and password are required"}), 400
    user = User.query.filter_by(username=username).first()
    if user and user.check_password(password):
        access_token = create_access_token(identity=username)
        return jsonify(access_token=access_token), 200
    else:
        return jsonify({"msg": "Invalid username or password"}), 401


@app.route('/register', methods=['POST'])
def register():
    if not request.is_json:
        return jsonify({"msg": "Missing JSON in request"}), 400
    data = request.get_json()
    username = data.get('username')
    password = data.get('password')
    if not username or not password:
        return jsonify({"msg": "Username and password are required"}), 400
    if User.query.filter_by(username=username).first():
        return jsonify({"msg": "Username already exists"}), 409
    new_user = User(username=username, password=password)
    db.session.add(new_user)
    db.session.commit()

    return jsonify({"msg": f"User '{username}' registered successfully"}), 201





"""if __name__ == '__main__':
    print("Server starting on 0.0.0.0:5000")
    serve(app, host='0.0.0.0', port=5000)"""
