# Mobile Deployment Examples

Complete code examples for deploying the trained model on Android, iOS, and other platforms.

---

## 🤖 Android Deployment (Java)

### 1. Add Dependencies to `build.gradle`

```gradle
dependencies {
    // PyTorch Android
    implementation 'org.pytorch:pytorch_android:1.10.0'
    implementation 'org.pytorch:pytorch_android_torchvision:1.10.0'
}
```

### 2. Place Model in Assets
- Copy `model_android.ptl` to `app/src/main/assets/`

### 3. Load and Run Model

```java
import org.pytorch.IValue;
import org.pytorch.Module;
import org.pytorch.Tensor;
import org.pytorch.torchvision.TensorImageUtils;
import android.graphics.Bitmap;

public class ObjectDetector {
    private Module model;
    private static final int INPUT_SIZE = 160;
    
    public ObjectDetector(Context context) {
        try {
            // Load model from assets
            model = Module.load(assetFilePath(context, "model_android.ptl"));
        } catch (IOException e) {
            Log.e("ObjectDetector", "Error loading model", e);
        }
    }
    
    public DetectionResult detect(Bitmap bitmap) {
        // Resize bitmap to 160x160
        Bitmap resizedBitmap = Bitmap.createScaledBitmap(
            bitmap, INPUT_SIZE, INPUT_SIZE, true
        );
        
        // Convert to tensor (normalized to [0, 1])
        Tensor inputTensor = TensorImageUtils.bitmapToFloat32Tensor(
            resizedBitmap,
            new float[]{0.0f, 0.0f, 0.0f},  // mean
            new float[]{1.0f, 1.0f, 1.0f}   // std
        );
        
        // Run inference
        IValue output = model.forward(IValue.from(inputTensor));
        
        // Parse output (boxes, labels, scores)
        // Output format: [{boxes: [...], labels: [...], scores: [...]}]
        Map<String, IValue> dict = output.toList()[0].toDictStringKey();
        
        float[] boxes = dict.get("boxes").toTensor().getDataAsFloatArray();
        long[] labels = dict.get("labels").toTensor().getDataAsLongArray();
        float[] scores = dict.get("scores").toTensor().getDataAsFloatArray();
        
        return new DetectionResult(boxes, labels, scores);
    }
    
    private String assetFilePath(Context context, String assetName) {
        File file = new File(context.getFilesDir(), assetName);
        if (file.exists() && file.length() > 0) {
            return file.getAbsolutePath();
        }
        
        try (InputStream is = context.getAssets().open(assetName)) {
            try (OutputStream os = new FileOutputStream(file)) {
                byte[] buffer = new byte[4 * 1024];
                int read;
                while ((read = is.read(buffer)) != -1) {
                    os.write(buffer, 0, read);
                }
                os.flush();
            }
            return file.getAbsolutePath();
        } catch (IOException e) {
            Log.e("ObjectDetector", "Error copying asset", e);
        }
        return null;
    }
}

class DetectionResult {
    float[] boxes;   // [N, 4] - x1, y1, x2, y2
    long[] labels;   // [N]
    float[] scores;  // [N]
    
    public DetectionResult(float[] boxes, long[] labels, float[] scores) {
        this.boxes = boxes;
        this.labels = labels;
        this.scores = scores;
    }
    
    public List<Detection> getDetections(float confidenceThreshold) {
        List<Detection> detections = new ArrayList<>();
        int numBoxes = scores.length;
        
        for (int i = 0; i < numBoxes; i++) {
            if (scores[i] >= confidenceThreshold) {
                float x1 = boxes[i * 4];
                float y1 = boxes[i * 4 + 1];
                float x2 = boxes[i * 4 + 2];
                float y2 = boxes[i * 4 + 3];
                
                detections.add(new Detection(
                    new RectF(x1, y1, x2, y2),
                    labels[i],
                    scores[i]
                ));
            }
        }
        return detections;
    }
}

class Detection {
    RectF box;
    long label;
    float score;
    
    public Detection(RectF box, long label, float score) {
        this.box = box;
        this.label = label;
        this.score = score;
    }
}
```

### 4. Usage Example

```java
// In your Activity or Fragment
ObjectDetector detector = new ObjectDetector(this);

// Capture image from camera
Bitmap image = captureImage();

// Run detection
DetectionResult result = detector.detect(image);

// Get filtered detections (confidence > 0.5)
List<Detection> detections = result.getDetections(0.5f);

// Draw boxes on image
for (Detection det : detections) {
    canvas.drawRect(det.box, paint);
    canvas.drawText(
        String.format("%.2f", det.score),
        det.box.left, det.box.top - 10,
        textPaint
    );
}
```

---

## 🍎 iOS Deployment (Swift)

### 1. Add Model to Xcode Project
- Drag `model_ios.mlmodel` into Xcode project
- Xcode automatically generates Swift interface

### 2. Load and Run Model

```swift
import UIKit
import CoreML
import Vision

class ObjectDetector {
    private var model: VNCoreMLModel?
    private let inputSize: CGFloat = 160
    
    init() {
        do {
            // Load CoreML model (auto-generated class)
            let mlModel = try model_ios(configuration: MLModelConfiguration())
            model = try VNCoreMLModel(for: mlModel.model)
        } catch {
            print("Error loading model: \(error)")
        }
    }
    
    func detect(image: UIImage, completion: @escaping ([Detection]) -> Void) {
        guard let model = model,
              let ciImage = CIImage(image: image) else {
            completion([])
            return
        }
        
        // Create request
        let request = VNCoreMLRequest(model: model) { request, error in
            guard let results = request.results as? [VNRecognizedObjectObservation] else {
                completion([])
                return
            }
            
            // Convert to Detection objects
            let detections = results.compactMap { observation -> Detection? in
                guard let label = observation.labels.first else { return nil }
                
                return Detection(
                    box: observation.boundingBox,
                    label: label.identifier,
                    score: observation.confidence
                )
            }
            
            completion(detections)
        }
        
        // Configure request
        request.imageCropAndScaleOption = .scaleFit
        
        // Run inference
        let handler = VNImageRequestHandler(ciImage: ciImage, options: [:])
        DispatchQueue.global(qos: .userInitiated).async {
            do {
                try handler.perform([request])
            } catch {
                print("Error performing inference: \(error)")
                completion([])
            }
        }
    }
}

struct Detection {
    let box: CGRect          // Normalized coordinates [0, 1]
    let label: String
    let score: Float
    
    // Convert normalized box to image coordinates
    func scaledBox(imageSize: CGSize) -> CGRect {
        return CGRect(
            x: box.origin.x * imageSize.width,
            y: (1 - box.origin.y - box.height) * imageSize.height,  // Flip Y
            width: box.width * imageSize.width,
            height: box.height * imageSize.height
        )
    }
}
```

### 3. Usage Example

```swift
// In your ViewController
class CameraViewController: UIViewController {
    let detector = ObjectDetector()
    
    func processImage(_ image: UIImage) {
        detector.detect(image: image) { detections in
            DispatchQueue.main.async {
                self.drawDetections(detections, on: image)
            }
        }
    }
    
    func drawDetections(_ detections: [Detection], on image: UIImage) {
        UIGraphicsBeginImageContextWithOptions(image.size, false, image.scale)
        image.draw(at: .zero)
        
        let context = UIGraphicsGetCurrentContext()
        context?.setStrokeColor(UIColor.red.cgColor)
        context?.setLineWidth(3.0)
        
        for detection in detections {
            guard detection.score >= 0.5 else { continue }
            
            // Draw box
            let scaledBox = detection.scaledBox(imageSize: image.size)
            context?.stroke(scaledBox)
            
            // Draw label
            let text = String(format: "%.2f", detection.score)
            let attributes: [NSAttributedString.Key: Any] = [
                .font: UIFont.boldSystemFont(ofSize: 16),
                .foregroundColor: UIColor.red
            ]
            text.draw(at: scaledBox.origin, withAttributes: attributes)
        }
        
        let resultImage = UIGraphicsGetImageFromCurrentImageContext()
        UIGraphicsEndImageContext()
        
        imageView.image = resultImage
    }
}
```

### 4. Optimized Real-Time Detection (Using AVFoundation)

```swift
import AVFoundation
import Vision

class RealtimeDetector: NSObject, AVCaptureVideoDataOutputSampleBufferDelegate {
    private var model: VNCoreMLModel?
    private let detectionQueue = DispatchQueue(label: "com.app.detection")
    
    override init() {
        super.init()
        setupModel()
    }
    
    private func setupModel() {
        do {
            let mlModel = try model_ios(configuration: MLModelConfiguration())
            model = try VNCoreMLModel(for: mlModel.model)
        } catch {
            print("Error loading model: \(error)")
        }
    }
    
    func captureOutput(
        _ output: AVCaptureOutput,
        didOutput sampleBuffer: CMSampleBuffer,
        from connection: AVCaptureConnection
    ) {
        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer),
              let model = model else { return }
        
        let request = VNCoreMLRequest(model: model) { [weak self] request, error in
            self?.processDetections(request.results as? [VNRecognizedObjectObservation])
        }
        
        request.imageCropAndScaleOption = .scaleFit
        
        let handler = VNImageRequestHandler(
            cvPixelBuffer: pixelBuffer,
            orientation: .up,
            options: [:]
        )
        
        detectionQueue.async {
            try? handler.perform([request])
        }
    }
    
    private func processDetections(_ observations: [VNRecognizedObjectObservation]?) {
        guard let observations = observations else { return }
        
        let detections = observations
            .filter { $0.confidence >= 0.5 }
            .map { Detection(
                box: $0.boundingBox,
                label: $0.labels.first?.identifier ?? "Unknown",
                score: $0.confidence
            )}
        
        DispatchQueue.main.async {
            // Update UI with detections
            self.updateUI(with: detections)
        }
    }
    
    private func updateUI(with detections: [Detection]) {
        // Update overlay, labels, etc.
    }
}
```

---

## 🌐 Python Deployment (ONNX Runtime)

### 1. Install Dependencies

```bash
pip install onnxruntime opencv-python numpy
```

### 2. Load and Run Model

```python
import onnxruntime as ort
import numpy as np
import cv2

class ObjectDetector:
    def __init__(self, model_path: str, input_size: int = 160):
        self.input_size = input_size
        self.session = ort.InferenceSession(
            model_path,
            providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
        )
        
        # Get input/output names
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [o.name for o in self.session.get_outputs()]
        
    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for model input"""
        # Resize
        img = cv2.resize(image, (self.input_size, self.input_size))
        
        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Normalize to [0, 1]
        img = img.astype(np.float32) / 255.0
        
        # Transpose to CHW format and add batch dimension
        img = np.transpose(img, (2, 0, 1))
        img = np.expand_dims(img, 0)
        
        return img
    
    def detect(
        self,
        image: np.ndarray,
        confidence_threshold: float = 0.5
    ) -> list:
        """Run object detection on image"""
        # Preprocess
        input_tensor = self.preprocess(image)
        
        # Run inference
        outputs = self.session.run(
            self.output_names,
            {self.input_name: input_tensor}
        )
        
        # Parse outputs (depends on model output format)
        # Assuming outputs = [boxes, labels, scores]
        boxes = outputs[0][0]     # [N, 4]
        labels = outputs[1][0]    # [N]
        scores = outputs[2][0]    # [N]
        
        # Filter by confidence
        mask = scores >= confidence_threshold
        
        detections = []
        for box, label, score in zip(boxes[mask], labels[mask], scores[mask]):
            detections.append({
                'box': box.tolist(),      # [x1, y1, x2, y2]
                'label': int(label),
                'score': float(score)
            })
        
        return detections
    
    def visualize(
        self,
        image: np.ndarray,
        detections: list,
        class_names: list = None
    ) -> np.ndarray:
        """Draw detections on image"""
        img = image.copy()
        h, w = img.shape[:2]
        
        # Scale factor from model input to original image
        scale_x = w / self.input_size
        scale_y = h / self.input_size
        
        for det in detections:
            x1, y1, x2, y2 = det['box']
            
            # Scale coordinates
            x1 = int(x1 * scale_x)
            y1 = int(y1 * scale_y)
            x2 = int(x2 * scale_x)
            y2 = int(y2 * scale_y)
            
            # Draw box
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw label
            label = det['label']
            score = det['score']
            
            if class_names and label < len(class_names):
                text = f"{class_names[label]} {score:.2f}"
            else:
                text = f"Class {label} {score:.2f}"
            
            cv2.putText(
                img, text, (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2
            )
        
        return img

# Usage example
if __name__ == "__main__":
    # Initialize detector
    detector = ObjectDetector("model_onnx_simplified.onnx")
    
    # Load image
    image = cv2.imread("test_image.jpg")
    
    # Run detection
    detections = detector.detect(image, confidence_threshold=0.5)
    
    # Visualize
    result = detector.visualize(
        image,
        detections,
        class_names=["background", "nameplate"]
    )
    
    # Display
    cv2.imshow("Detections", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # Print results
    print(f"Found {len(detections)} objects:")
    for i, det in enumerate(detections):
        print(f"  {i+1}. Label: {det['label']}, "
              f"Score: {det['score']:.3f}, "
              f"Box: {det['box']}")
```

### 3. Batch Processing

```python
import glob
from pathlib import Path

def batch_process(
    input_dir: str,
    output_dir: str,
    model_path: str,
    confidence: float = 0.5
):
    """Process all images in directory"""
    detector = ObjectDetector(model_path)
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    image_files = glob.glob(f"{input_dir}/*.jpg") + \
                  glob.glob(f"{input_dir}/*.png")
    
    for img_path in image_files:
        print(f"Processing: {img_path}")
        
        # Load and detect
        image = cv2.imread(img_path)
        detections = detector.detect(image, confidence)
        
        # Visualize and save
        result = detector.visualize(image, detections)
        
        output_path = Path(output_dir) / Path(img_path).name
        cv2.imwrite(str(output_path), result)
        
        print(f"  Found {len(detections)} objects")

# Usage
batch_process(
    input_dir="test_images",
    output_dir="results",
    model_path="model_onnx_simplified.onnx",
    confidence=0.5
)
```

---

## 🔥 Flask API Server

Complete REST API for model serving:

```python
from flask import Flask, request, jsonify
import cv2
import numpy as np
import base64
from io import BytesIO
from PIL import Image

app = Flask(__name__)
detector = ObjectDetector("model_onnx_simplified.onnx")

@app.route('/detect', methods=['POST'])
def detect():
    """
    POST /detect
    Body: { "image": "<base64_encoded_image>" }
    Returns: { "detections": [...] }
    """
    try:
        # Get image from request
        data = request.get_json()
        image_b64 = data['image']
        
        # Decode base64 image
        image_bytes = base64.b64decode(image_b64)
        image = Image.open(BytesIO(image_bytes))
        image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Run detection
        detections = detector.detect(image, confidence_threshold=0.5)
        
        return jsonify({
            'success': True,
            'detections': detections,
            'count': len(detections)
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({'status': 'healthy'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
```

### API Usage

```bash
# Start server
python api_server.py

# Test with curl
curl -X POST http://localhost:5000/detect \
  -H "Content-Type: application/json" \
  -d '{"image": "<base64_image>"}'
```

---

## 📊 Performance Benchmarking

```python
import time
import numpy as np

def benchmark_model(model_path: str, num_runs: int = 100):
    """Benchmark model inference time"""
    detector = ObjectDetector(model_path)
    
    # Create dummy input
    dummy_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    # Warmup
    for _ in range(10):
        _ = detector.detect(dummy_image)
    
    # Benchmark
    times = []
    for _ in range(num_runs):
        start = time.perf_counter()
        _ = detector.detect(dummy_image)
        end = time.perf_counter()
        times.append((end - start) * 1000)  # Convert to ms
    
    # Statistics
    times = np.array(times)
    print(f"Inference Time Statistics ({num_runs} runs):")
    print(f"  Mean:   {times.mean():.2f} ms")
    print(f"  Median: {np.median(times):.2f} ms")
    print(f"  Std:    {times.std():.2f} ms")
    print(f"  Min:    {times.min():.2f} ms")
    print(f"  Max:    {times.max():.2f} ms")
    print(f"  FPS:    {1000/times.mean():.2f}")

# Run benchmark
benchmark_model("model_onnx_simplified.onnx")
```

---

## 🚀 Docker Deployment

```dockerfile
FROM python:3.9-slim

# Install dependencies
RUN pip install --no-cache-dir \
    onnxruntime \
    opencv-python-headless \
    numpy \
    flask

# Copy model and code
COPY model_onnx_simplified.onnx /app/model.onnx
COPY api_server.py /app/api_server.py

WORKDIR /app

# Expose port
EXPOSE 5000

# Run server
CMD ["python", "api_server.py"]
```

```bash
# Build and run
docker build -t object-detector .
docker run -p 5000:5000 object-detector
```

---

## 📝 Summary

This guide provides complete deployment examples for:
- ✅ **Android** (Java) - TorchScript Mobile
- ✅ **iOS** (Swift) - CoreML
- ✅ **Python** (ONNX Runtime) - Universal deployment
- ✅ **Flask API** - REST API server
- ✅ **Docker** - Containerized deployment
- ✅ **Benchmarking** - Performance measurement

All examples are production-ready and can be directly integrated into your applications!
