#ifndef _L_SHAPED_FIT_H
#define _L_SHAPED_FIT_H

#include <vector>
#include <cmath>
#include <algorithm>
#include <limits>
#include <numeric>
#include <cassert>

namespace geo {
// ---------------------
// Custom Point2f
// ---------------------
struct Point2f {
    double x;
    double y;
    Point2f() : x(0), y(0) {}
    Point2f(double _x, double _y) : x(_x), y(_y) {}
};

// ---------------------
// Custom RotatedRect
// ---------------------
struct RotatedRect {
    Point2f center;
    Point2f size;   // width, height
    double angle;   // radians
    bool valid;

    RotatedRect() : center(), size(), angle(0), valid(false) {}
    RotatedRect(Point2f c, Point2f s, double a)
        : center(c), size(s), angle(a), valid(true) {}
};

// ---------------------
// Lightweight Mat
// ---------------------
class Mat {
public:
    int rows, cols;
    std::vector<double> data;

    Mat() : rows(0), cols(0) {}

    Mat(int r, int c, double value = 0.0)
        : rows(r), cols(c), data(r * c, value) {}

    static Mat zeros(int r, int c) {
        return Mat(r, c, 0.0);
    }

    inline double& at(int r, int c) {
        return data[r * cols + c];
    }

    inline const double& at(int r, int c) const {
        return data[r * cols + c];
    }

    Mat t() const {
        Mat res(cols, rows);
        for (int r = 0; r < rows; r++)
            for (int c = 0; c < cols; c++)
                res.at(c, r) = at(r, c);
        return res;
    }
};

// ---------------------
// LShapedFIT class
// ---------------------
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

    void calc_cross_point(const double a0, const double a1,
                          const double b0, const double b1,
                          const double c0, const double c1,
                          double& x, double& y);

    RotatedRect calc_rect_contour();
    RotatedRect build_rotated_rect_from_vertices(const std::vector<Point2f>& pts);

    // Helpers
    void minMax(const std::vector<double>& v, double& mn, double& mx);
};
} // namespace geo
#endif // _L_SHAPED_FITTING_H
