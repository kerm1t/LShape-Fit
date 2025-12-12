#ifndef _L_SHAPED_FIT_H
#define _L_SHAPED_FIT_H

#pragma once
#include <vector>
#include <cmath>
#include <algorithm>
#include <limits>
#include <numeric>
#include <cassert>

namespace geo {
struct Point2f {
    double x, y;
    Point2f() : x(0), y(0) {}
    Point2f(double xx, double yy) : x(xx), y(yy) {}
};

struct RotatedRect {
    Point2f center;
    Point2f size;   // width, height
    double angle;   // radians
    bool valid;

    RotatedRect() : angle(0), valid(false) {}
    RotatedRect(Point2f c, Point2f s, double a)
        : center(c), size(s), angle(a), valid(true) {}
};

class Mat {
public:
    int rows, cols;
    std::vector<double> data;

    Mat() : rows(0), cols(0) {}
    Mat(int r, int c, double val = 0.0)
        : rows(r), cols(c), data(r * c, val) {}

    static Mat zeros(int r, int c) { return Mat(r, c, 0.0); }

    inline double& at(int r, int c) {
        return data[r * cols + c];
    }
    inline const double& at(int r, int c) const {
        return data[r * cols + c];
    }
};

class LShapedFIT {
public:
    enum Criterion { AREA, NEAREST, VARIANCE };

    LShapedFIT();
    ~LShapedFIT();

    RotatedRect FitBox(std::vector<Point2f>* pointcloud_ptr);
    std::vector<Point2f> getRectVertex();

private:
    double min_dist_of_nearest_crit_;
    double dtheta_deg_for_search_;
    Criterion criterion_;

    std::vector<Point2f> vertex_pts_;

    std::vector<double> a_;
    std::vector<double> b_;
    std::vector<double> c_;

    double calc_area_criterion(const Mat& c1, const Mat& c2);
    double calc_nearest_criterion(const Mat& c1, const Mat& c2);
    double calc_variances_criterion(const Mat& c1, const Mat& c2);

    double calc_var(const std::vector<double>& v);

    void calc_cross_point(double a0, double a1, double b0, double b1,
                          double c0, double c1, double& x, double& y);

    RotatedRect calc_rect_contour();
    RotatedRect build_rotated_rect_from_vertices(std::vector<Point2f>& pts);

    void minMax(const std::vector<double>& v, double& mn, double& mx);
};

} // namespace geo
#endif // _L_SHAPED_FITTING_H
