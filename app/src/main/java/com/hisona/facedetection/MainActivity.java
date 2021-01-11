package com.hisona.facedetection;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;
import androidx.camera.core.AspectRatio;
import androidx.camera.core.Camera;
import androidx.camera.core.CameraSelector;
import androidx.camera.core.ImageAnalysis;
import androidx.camera.core.ImageProxy;
import androidx.camera.core.Preview;
import androidx.camera.lifecycle.ProcessCameraProvider;
import androidx.camera.view.PreviewView;
import androidx.constraintlayout.widget.ConstraintLayout;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;
import androidx.lifecycle.LifecycleOwner;

import android.Manifest;
import android.annotation.SuppressLint;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.media.Image;
import android.os.Bundle;
import android.os.SystemClock;
import android.util.DisplayMetrics;
import android.util.Log;
import android.util.Size;
import android.view.Surface;
import android.view.View;
import android.view.ViewGroup;
import android.widget.ImageButton;
import android.widget.TextView;
import android.widget.Toast;

import com.google.common.util.concurrent.ListenableFuture;

import org.pytorch.Module;

import java.util.Locale;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.Executor;
import java.util.concurrent.Executors;

public class MainActivity extends AppCompatActivity {

    private static final String TAG = MainActivity.class.getSimpleName();

    private static final int REQUEST_CODE_CAMERA_PERMISSION = 200;
    private static final String[] PERMISSIONS = {Manifest.permission.CAMERA};

    ConstraintLayout mContainer;
    TextView mTextView;
    PreviewView mViewFinder;
    ImageButton mCameraCaptureButton;
    View mBoxPrediction;

    Executor mExecutor;
    Module mModule;
    private boolean isFront;


    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_camera);

        mContainer = findViewById(R.id.camera_container);
        mTextView = findViewById(R.id.text_prediction);
        mCameraCaptureButton = findViewById(R.id.camera_capture_button);
        mViewFinder = findViewById(R.id.view_finder);
        mBoxPrediction = findViewById(R.id.box_prediction);

        mBoxPrediction.setVisibility(View.VISIBLE);

        mExecutor = Executors.newSingleThreadExecutor();

        isFront = false;

        mCameraCaptureButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                isFront = !isFront;
                startCamera(isFront);
            }
        });

        if(checkPermission()) {
            startCamera(isFront);
        }

        mModule = Module.load(FaceUtils.assetFilePath(this, "mbv2.pt"));

    }

    private boolean checkPermission() {
        if (ActivityCompat.checkSelfPermission(this, Manifest.permission.CAMERA)
                != PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(
                    this,
                    PERMISSIONS,
                    REQUEST_CODE_CAMERA_PERMISSION);
            return false;
        }
        return true;
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, String[] permissions, int[] grantResults) {
        if (requestCode == REQUEST_CODE_CAMERA_PERMISSION) {
            if (grantResults[0] == PackageManager.PERMISSION_DENIED) {
                Toast.makeText(
                        this,
                        "You can't use image classification example without granting CAMERA permission",
                        Toast.LENGTH_LONG)
                        .show();
                finish();
            } else {
                startCamera(isFront);
            }
        }
    }

    private void startCamera(boolean isFront) {

        ListenableFuture<ProcessCameraProvider> cameraProviderFuture
                = ProcessCameraProvider.getInstance(this);

        cameraProviderFuture.addListener(new Runnable() {
            @Override
            public void run() {
                try {
                    ProcessCameraProvider cameraProvider = cameraProviderFuture.get();
                    bindPreview(cameraProvider, isFront);
                } catch (ExecutionException | InterruptedException e) {
                    // No errors need to be handled for this Future.
                    // This should never be reached.
                }
            }
        }, ContextCompat.getMainExecutor(this));
    }

    void bindPreview(@NonNull ProcessCameraProvider cameraProvider, boolean isFront) {

        Preview preview = new Preview.Builder()
                .setTargetAspectRatio(AspectRatio.RATIO_4_3)
                .build();

        CameraSelector cameraSelector = new CameraSelector.Builder()
                .requireLensFacing(isFront ? CameraSelector.LENS_FACING_FRONT : CameraSelector.LENS_FACING_BACK)
                .build();

        mExecutor = Executors.newSingleThreadExecutor();

        ImageAnalysis imageAnalysis = new ImageAnalysis.Builder()
                .setTargetResolution(new Size(480, 480))
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .build();

        FaceBox[] mAnchors = FaceUtils.getAnchors();

        imageAnalysis.setAnalyzer(mExecutor, new ImageAnalysis.Analyzer() {
            @Override
            public void analyze(@NonNull ImageProxy image) {
                int rotationDegrees = image.getImageInfo().getRotationDegrees();

                @SuppressLint("UnsafeExperimentalUsageError")
                Bitmap bitmap = FaceUtils.preProcessing(FaceUtils.imageToBitmap(image.getImage()),
                        rotationDegrees, isFront);
                Prediction predict = FaceUtils.runningModel(mModule, mAnchors, bitmap);

                runOnUiThread(new Runnable() {
                    @Override
                    public void run() {

                        float fps;
                        if(predict.elapse > 0)
                            fps = 1000.f / predict.elapse;
                        else
                            fps = 1000.f;

                        String str = String.format(Locale.US, "%1.3f, %3.1f fps, %d ms",
                                predict.score, fps, predict.elapse);

                        mTextView.setText(str);

                        ViewGroup.MarginLayoutParams params = (ViewGroup.MarginLayoutParams)mBoxPrediction.getLayoutParams();

                        int max_h = mViewFinder.getHeight();
                        int margin_x = (max_h - mViewFinder.getWidth()) / 2;

                        params.leftMargin = (predict.x1 * max_h) / FaceUtils.IMAGE_WIDTH - margin_x;
                        params.topMargin = (predict.y1 * max_h) / FaceUtils.IMAGE_HEIGHT;
                        params.width = (predict.x2 - predict.x1) * max_h / FaceUtils.IMAGE_WIDTH;
                        params.height = (predict.y2 - predict.y1) * max_h / FaceUtils.IMAGE_HEIGHT;

                        mBoxPrediction.setLayoutParams(params);
                        mBoxPrediction.setVisibility(View.VISIBLE);

                        /*
                        Canvas canvas = new Canvas(bitmap);
                        Paint paint = new Paint();
                        paint.setColor(Color.RED);
                        paint.setStyle(Paint.Style.STROKE);
                        paint.setStrokeWidth(3);

                        canvas.drawRect(predict.x1, predict.y1, predict.x2, predict.y2, paint);
                        canvas.drawRect(2, 2, 638, 638, paint);

                        mCameraCaptureButton.setImageBitmap(bitmap);
                        */
                    }
                });

                image.close();
            }
        });

        cameraProvider.unbindAll();
        Camera camera = cameraProvider.bindToLifecycle((LifecycleOwner)this,
                cameraSelector, imageAnalysis, preview);

        preview.setSurfaceProvider(mViewFinder.getSurfaceProvider());

    }

}