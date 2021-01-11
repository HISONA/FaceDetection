package com.hisona.facedetection;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.ImageFormat;
import android.graphics.Matrix;
import android.graphics.Paint;
import android.graphics.Rect;
import android.graphics.YuvImage;
import android.media.Image;
import android.os.SystemClock;
import android.util.Log;

import org.pytorch.IValue;
import org.pytorch.Module;
import org.pytorch.Tensor;

import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.nio.ByteBuffer;
import java.nio.FloatBuffer;


public class FaceUtils {
    private static final String TAG = FaceUtils.class.getSimpleName();

    public static final int IMAGE_WIDTH = 640;
    public static final int IMAGE_HEIGHT = 640;

    /**
     * Copies specified asset to the file in /files app directory and returns this file absolute path.
     *
     * @return absolute file path
     */
    public static String assetFilePath(Context context, String assetName) {
        File file = new File(context.getFilesDir(), assetName);

        if (file.exists() && file.length() > 0) {
            return file.getAbsolutePath();
        }

        try {
            InputStream is = context.getAssets().open(assetName);
            OutputStream os = new FileOutputStream(file);

            byte[] buffer = new byte[4 * 1024];
            int read;

            while ((read = is.read(buffer)) != -1) {
                os.write(buffer, 0, read);
            }
            os.flush();

            return file.getAbsolutePath();
        } catch (Exception e) {
            return null;
        }
    }

    public static Bitmap imageToBitmap(Image image) {
        Image.Plane[] planes = image.getPlanes();

        ByteBuffer yBuffer = planes[0].getBuffer();
        ByteBuffer uBuffer = planes[1].getBuffer();
        ByteBuffer vBuffer = planes[2].getBuffer();

        int ySize = yBuffer.remaining();
        int uSize = uBuffer.remaining();
        int vSize = vBuffer.remaining();

        byte[] nv21 = new byte[ySize + uSize + vSize];

        //U and V are swapped
        yBuffer.get(nv21, 0, ySize);
        vBuffer.get(nv21, ySize, vSize);
        uBuffer.get(nv21, ySize + vSize, uSize);

        YuvImage yuvImage = new YuvImage(nv21, ImageFormat.NV21, image.getWidth(), image.getHeight(), null);
        ByteArrayOutputStream out = new ByteArrayOutputStream();

        yuvImage.compressToJpeg(new Rect(0, 0, yuvImage.getWidth(), yuvImage.getHeight()), 75, out);

        byte[] imageBytes = out.toByteArray();
        return BitmapFactory.decodeByteArray(imageBytes, 0, imageBytes.length);
    }

    public static Tensor bitmapToFloat32Tensor(
            final Bitmap bitmap, final float[] normMeanRGB, final float normStdRGB[]) {
        return bitmapToFloat32Tensor(
                bitmap, 0, 0, bitmap.getWidth(), bitmap.getHeight(), normMeanRGB, normStdRGB);
    }

    public static Tensor bitmapToFloat32Tensor(
            final Bitmap bitmap,
            int x,
            int y,
            int width,
            int height,
            float[] normMeanRGB,
            float[] normStdRGB) {

        final FloatBuffer floatBuffer = Tensor.allocateFloatBuffer(3 * width * height);
        bitmapToFloatBuffer(bitmap, x, y, width, height, normMeanRGB, normStdRGB, floatBuffer, 0);

        return Tensor.fromBlob(floatBuffer, new long[]{1, 3, height, width});
    }

    public static void bitmapToFloatBuffer(
            final Bitmap bitmap,
            final int x,
            final int y,
            final int width,
            final int height,
            final float[] normMeanRGB,
            final float[] normStdRGB,
            final FloatBuffer outBuffer,
            final int outBufferOffset) {

        checkOutBufferCapacity(outBuffer, outBufferOffset, width, height);

        final int pixelsCount = height * width;
        final int[] pixels = new int[pixelsCount];

        bitmap.getPixels(pixels, 0, width, x, y, width, height);

        final int offset_g = pixelsCount;
        final int offset_b = 2 * pixelsCount;

        for (int i = 0; i < pixelsCount; i++) {
            final int c = pixels[i];
            float r = ((c >> 16) & 0xff) / 1.0f;
            float g = ((c >> 8) & 0xff) / 1.0f;
            float b = ((c) & 0xff) / 1.0f;

            // System.out.print(" "+r+" ;"+g+" ;"+b);

            float rF = (r - normMeanRGB[0]) / normStdRGB[0];
            float gF = (g - normMeanRGB[1]) / normStdRGB[1];
            float bF = (b - normMeanRGB[2]) / normStdRGB[2];

            outBuffer.put(outBufferOffset + i, rF);
            outBuffer.put(outBufferOffset + offset_g + i, gF);
            outBuffer.put(outBufferOffset + offset_b + i, bF);
        }
    }

    private static void checkOutBufferCapacity(FloatBuffer outBuffer, int outBufferOffset, int tensorWidth, int tensorHeight) {
        if (outBufferOffset + 3 * tensorWidth * tensorHeight > outBuffer.capacity()) {
            throw new IllegalStateException("Buffer underflow");
        }
    }

    public static Bitmap preProcessing(Bitmap bitmap, int degrees, boolean flip)
    {
        int imgw = bitmap.getWidth();
        int imgh = bitmap.getHeight();
        float scaleHeight;
        float scaleWidth;

        if(imgw > imgh) {
            scaleHeight = (float)IMAGE_HEIGHT / (float)imgw;
            scaleWidth = (float)IMAGE_WIDTH / (float)imgw;
        } else {
            scaleWidth = (float)IMAGE_WIDTH / (float)imgh;
            scaleHeight = (float)IMAGE_HEIGHT / (float)imgh;
        }

        if(flip) scaleWidth = -scaleWidth;

        Matrix matrix = new Matrix();
        matrix.postRotate(degrees);

        matrix.postScale(scaleWidth, scaleHeight);

        // New image
        Bitmap newbm = Bitmap.createBitmap(bitmap, 0, 0, imgw, imgh, matrix,true);

        int imgmin = Math.min(newbm.getWidth(),newbm.getHeight());
        int padsize = (int) (IMAGE_WIDTH - imgmin)/2;

        Bitmap mergebitmap = Bitmap.createBitmap(640, 640, Bitmap.Config.ARGB_8888);

        if(newbm.getHeight() > newbm.getWidth()) {
            // Create a new blank bitmap
            Bitmap bgBitmap1 = Bitmap.createBitmap(padsize, IMAGE_HEIGHT, Bitmap.Config.ARGB_8888);
            Bitmap bgBitmap2 = Bitmap.createBitmap(IMAGE_WIDTH - padsize, IMAGE_HEIGHT, Bitmap.Config.ARGB_8888);
            Canvas canvasadd = new Canvas(mergebitmap);
            canvasadd.drawBitmap(bgBitmap1, 0, 0, null);
            canvasadd.drawBitmap(newbm, padsize, 0, null);
            canvasadd.drawBitmap(bgBitmap2, padsize + newbm.getWidth(), 0, null);
        } else if (newbm.getHeight() < newbm.getWidth()) {
            // Create a new blank bitmap
            Bitmap bgBitmap1 = Bitmap.createBitmap(IMAGE_WIDTH, padsize, Bitmap.Config.ARGB_8888);
            Bitmap bgBitmap2 = Bitmap.createBitmap(IMAGE_WIDTH, IMAGE_HEIGHT - padsize, Bitmap.Config.ARGB_8888);
            Canvas canvasadd = new Canvas(mergebitmap);
            canvasadd.drawBitmap(bgBitmap1, 0, 0, null);
            canvasadd.drawBitmap(newbm, 0, padsize, null);
            canvasadd.drawBitmap(bgBitmap2, 0, padsize + newbm.getWidth(),  null);
        } else {
            mergebitmap = newbm;
        }

        return mergebitmap;
    }

    public static float intersectionOverUnion(int rect1x, int rect1y, int rect1w, int rect1h,
                                              int rect2x, int rect2y, int rect2w, int rect2h) {

        int leftColumnMax = Math.max(rect1x, rect2x);
        int rightColumnMin = Math.min(rect1w,rect2w);
        int upRowMax = Math.max(rect1y, rect2y);
        int downRowMin = Math.min(rect1h,rect2h);

        if (leftColumnMax>=rightColumnMin || downRowMin<=upRowMax){
            return 0;
        }

        int s1 = (rect1w-rect1x)*(rect1h-rect1y);
        int s2 = (rect2w-rect2x)*(rect2h-rect2y);
        float sCross = (downRowMin-upRowMax)*(rightColumnMin-leftColumnMax);

        return sCross/(s1+s2-sCross);
    }

    public static FaceBox[] getAnchors() {

        int num = 0;

        int imw = IMAGE_WIDTH;
        int imh = IMAGE_HEIGHT;

        double fmw1 = Math.ceil(((float) imw) / 16.0f );
        double fmh1 = Math.ceil(((float) imh) / 16.0f );
        double fmw2 = Math.ceil(((float) imw) / 32.0f );
        double fmh2 = Math.ceil(((float) imh) / 32.0f );
        double fmw3 = Math.ceil(((float) imw) / 64.0f );
        double fmh3 = Math.ceil(((float) imh) / 64.0f );

        int totalnum = 2*(((int)fmh1)*((int)fmw1)+((int)fmh2)*((int)fmw2)+((int)fmh3)*((int)fmw3));

        FaceBox[] Anchors = new FaceBox[totalnum];

        for (int k = 0; k < fmh1; k++) {
            for (int j = 0; j < fmw1; j++) {
                Anchors[num] = new FaceBox();
                Anchors[num].x1 = (float)((j + 0.5) * 16 / (double)imw);
                Anchors[num].y1 = (float)((k + 0.5) * 16 / (double)imh);
                Anchors[num].x2 = (float)(16.0 / (double)imw);
                Anchors[num].y2 = (float)(16.0 / (double)imh);
                num += 1;

                Anchors[num] = new FaceBox();
                Anchors[num].x1 = (float)((j + 0.5) * 16 / (double)imw);
                Anchors[num].y1 = (float)((k + 0.5) * 16 / (double)imh);
                Anchors[num].x2 = (float)(32.0 / (double)imw);
                Anchors[num].y2 = (float)(32.0 / (double)imh);
                num += 1;
            }
        }

        for (int k = 0; k < fmh2; k++) {
            for (int j = 0; j < fmw2; j++) {
                Anchors[num] = new FaceBox();
                Anchors[num].x1 = (float)((j + 0.5) * 32 / (double)imw);
                Anchors[num].y1 = (float)((k + 0.5) * 32 / (double)imh);
                Anchors[num].x2 = (float)(64.0 / (double)imw);
                Anchors[num].y2 = (float)(64.0 / (double)imh);
                num += 1;

                Anchors[num] = new FaceBox();
                Anchors[num].x1 = (float)((j + 0.5) * 32 / (double)imw);
                Anchors[num].y1 = (float)((k + 0.5) * 32 / (double)imh);
                Anchors[num].x2 = (float)(128.0 / (double)imw);
                Anchors[num].y2 = (float)(128.0 / (double)imh);
                num += 1;
            }
        }

        for (int k = 0; k < fmh3; k++) {
            for (int j = 0; j < fmw3; j++) {
                Anchors[num] = new FaceBox();
                Anchors[num].x1 = (float)((j + 0.5) * 64 / (double)imw);
                Anchors[num].y1 = (float)((k + 0.5) * 64 / (double)imh);
                Anchors[num].x2 = (float)(256.0 / (double)imw);
                Anchors[num].y2 = (float)(256.0 / (double)imh);
                num += 1;

                Anchors[num] = new FaceBox();
                Anchors[num].x1 = (float)((j + 0.5) * 64 / ((double) imw));
                Anchors[num].y1 = (float)((k + 0.5) * 64 / ((double) imh));
                Anchors[num].x2 = (float)(512.0 / ((double) imw));
                Anchors[num].y2 = (float)(512.0 / ((double) imh));
                num += 1;
            }
        }
        return Anchors;
    }

    public static Prediction runningModel(Module module, FaceBox[] anchors, Bitmap bitmap) {

        // prepareInputTensor
        float[] face_mean = new float[]{116.0f, 117.0f, 111.0f};   //offset to {104.0f, 117.0f, 123.0f}
        float[] face_std = new float[]{1.0f, 1.0f, 1.0f};
        final Tensor inputTensor = bitmapToFloat32Tensor(bitmap, face_mean, face_std);

        // Log.e(TAG, "input length: " + inputTensor.getDataAsFloatArray().length);
        // Log.e(TAG, "inputTensor: " +inputTensor.numel());

        // running the model
        long startTime = SystemClock.elapsedRealtime();
        final IValue[] outputTensor = module.forward(IValue.from(inputTensor)).toTuple();
        long endTime = SystemClock.elapsedRealtime();

        long infTime = endTime - startTime;

        // Log.e(TAG, "time ellipse：" + infTime + "ms");

        //*************************** bbox ******************************//
        float[] facebox = outputTensor[0].toTensor().getDataAsFloatArray();
        float[] facecls = outputTensor[1].toTensor().getDataAsFloatArray();
        float[] faceldm = outputTensor[2].toTensor().getDataAsFloatArray();

        // Log.e(TAG,"face box length : " + facebox.length);
        // Log.e(TAG,"face cls length : " + facecls.length);
        // Log.e(TAG, "face landmark length : " + faceldm.length);

        int imw = bitmap.getWidth();
        int imh = bitmap.getHeight();

        double fmw1 = Math.ceil((float)imw / 16.0f);
        double fmh1 = Math.ceil((float)imh / 16.0f);
        double fmw2 = Math.ceil((float)imw / 32.0f);
        double fmh2 = Math.ceil((float)imh / 32.0f);
        double fmw3 = Math.ceil((float)imw / 64.0f);
        double fmh3 = Math.ceil((float)imh / 64.0f);

        int totalnum = 2*(((int)fmh1)*((int)fmw1)+((int)fmh2)*((int)fmw2)+((int)fmh3)*((int)fmw3));
        float maxcls = 0.0f;
        float maxx = 0.0f;
        float maxy = 0.0f;
        float maxw = 0.0f;
        float maxh = 0.0f;
        float[] faceconf = new float[totalnum];
        int[] faceidx = new int[totalnum];
        int clsnum = 0;
        int maxidx = 0;

        FaceBox[] faceBoxes = new FaceBox[totalnum];

        for (int i = 0; i < facebox.length; i = i+4) {
            int clsidx = (int) (i/4);
            float clsconf = facecls[2*clsidx + 1];

            faceBoxes[clsidx] = new FaceBox();
            faceBoxes[clsidx].score = clsconf;
            faceBoxes[clsidx].x1 = facebox[i];
            faceBoxes[clsidx].y1 = facebox[i+1];
            faceBoxes[clsidx].x2 = facebox[i+2];
            faceBoxes[clsidx].y2 = facebox[i+3];

            if(clsconf > 0.2) {
                faceconf[clsnum] = clsconf;
                faceidx[clsnum] = clsidx;
                clsnum +=1;
            }

            if (clsconf > maxcls) {
                maxcls = clsconf;
                maxidx = clsidx;
            }
        }

        int x1 = 0, y1 = 0, x2 = 0, y2 = 0;

        if(clsnum > 0) {
            float[] PredConfOr = new float[clsnum];
            int[] PredBoxX1 = new int[clsnum];
            int[] PredBoxY1 = new int[clsnum];
            int[] PredBoxX2 = new int[clsnum];
            int[] PredBoxY2 = new int[clsnum];

            for (int k = 0; k < clsnum; k++) {
                for (int j = 0; j < clsnum - k - 1; j++) {
                    if (faceconf[j] < faceconf[j + 1]) {
                        float tmp = faceconf[j];
                        faceconf[j] = faceconf[j + 1];
                        faceconf[j + 1] = tmp;
                        int idxtmp = faceidx[j];
                        faceidx[j] = faceidx[j + 1];
                        faceidx[j + 1] = idxtmp;
                    }
                }
            }

            for (int k = 0; k < 1; k++) {
                double ax1 = anchors[faceidx[k]].x1;
                double ay1 = anchors[faceidx[k]].y1;
                double ax2 = anchors[faceidx[k]].x2;
                double ay2 = anchors[faceidx[k]].y2;

                double bbox_x1 = ax1 + maxx * 0.1 * ax2;
                double bbox_y1 = ay1 + maxy * 0.1 * ay2;
                double bbox_x2 = ax2 * Math.exp(maxw * 0.2);
                double bbox_y2 = ay2 * Math.exp(maxh * 0.2);
                float boxconf = faceconf[faceidx[k]];

                bbox_x1 = bbox_x1 - bbox_x2 / 2;
                bbox_y1 = bbox_y1 - bbox_y2 / 2;
                bbox_x2 = bbox_x2 + bbox_x1;
                bbox_y2 = bbox_y2 + bbox_y1;

                PredBoxX1[k] = (int) Math.round(bbox_x1 * IMAGE_WIDTH);
                PredBoxY1[k] = (int) Math.round(bbox_y1 * IMAGE_HEIGHT);
                PredBoxX2[k] = (int) Math.round(bbox_x2 * IMAGE_WIDTH);
                PredBoxY2[k] = (int) Math.round(bbox_y2 * IMAGE_HEIGHT);

                x1 = PredBoxX1[k];
                y1 = PredBoxY1[k];
                x2 = PredBoxX2[k];
                y2 = PredBoxY2[k];
            }
        }

        return new Prediction(maxcls, infTime, x1, y1, x2, y2);
    }

}
