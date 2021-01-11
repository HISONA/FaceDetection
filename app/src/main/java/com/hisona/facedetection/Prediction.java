package com.hisona.facedetection;

public class Prediction {
    float score;
    long elapse;
    int x1;
    int y1;
    int x2;
    int y2;

    Prediction(float score, long elapse, int x1, int y1, int x2, int y2) {
        this.score = score;
        this.elapse = elapse;
        this.x1 = x1;
        this.y1 = y1;
        this.x2 = x2;
        this.y2 = y2;
    }
}
