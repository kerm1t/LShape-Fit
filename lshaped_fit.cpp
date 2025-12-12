#include "lshaped_fit.h"
#include <iostream>

namespace geo {

LShapedFIT::LShapedFIT() {
    min_dist_of_nearest_crit_ = 0.01;
    dtheta_deg_for_search_    = 1.0;
    criterion_ = VARIANCE;
}

LShapedFIT::~LShapedFIT() {}

void LShapedFIT::minMax(const std::vector<double>& v, double& mn, double& mx)
{
    mn = std::numeric_limits<double>::max();
    mx = -mn;
    for (double d : v) {
        if (d < mn) mn = d;
        if (d > mx) mx = d;
    }
}

RotatedRect LShapedFIT::FitBox(std::vector<Point2f>* pointcloud_ptr)
{
    auto& pts = *pointcloud_ptr;
    if (pts.size() < 3) return RotatedRect();

    Mat M(pts.size(), 2);
    for (size_t i = 0; i < pts.size(); i++) {
        M.at(i,0) = pts[i].x;
        M.at(i,1) = pts[i].y;
    }

    double dtheta = dtheta_deg_for_search_ * M_PI / 180.0;
    double best_cost = -std::numeric_limits<double>::infinity();
    double best_theta = 0;

    int loop_number = std::ceil((M_PI/2 - dtheta) / dtheta);

    Mat c1(pts.size(), 1), c2(pts.size(), 1);

    for (int k = 0; k < loop_number; k++) {
        double theta = k * dtheta;
        if (theta >= M_PI/2 - dtheta) break;

        double c = std::cos(theta);
        double s = std::sin(theta);

        for (size_t i = 0; i < pts.size(); i++) {
            double x = M.at(i,0), y = M.at(i,1);
            c1.at(i,0) = x*c + y*s;
            c2.at(i,0) = -x*s + y*c;
        }

        double cost = 0;
        switch(criterion_) {
            case AREA:    cost = calc_area_criterion(c1,c2); break;
            case NEAREST: cost = calc_nearest_criterion(c1,c2); break;
            case VARIANCE:cost = calc_variances_criterion(c1,c2); break;
        }

        if (cost > best_cost) {
            best_cost = cost;
            best_theta = theta;
        }
    }

    // Recompute projections at best_theta
    double c = std::cos(best_theta);
    double s = std::sin(best_theta);

    for (size_t i = 0; i < pts.size(); i++) {
        double x = M.at(i,0), y = M.at(i,1);
        c1.at(i,0) = x*c + y*s;
        c2.at(i,0) = -x*s + y*c;
    }

    std::vector<double> v1(pts.size()), v2(pts.size());
    for (size_t i = 0; i < pts.size(); i++) {
        v1[i] = c1.at(i,0);
        v2[i] = c2.at(i,0);
    }

    double mn1, mx1, mn2, mx2;
    minMax(v1, mn1, mx1);
    minMax(v2, mn2, mx2);

    a_.clear(); b_.clear(); c_.clear();
    a_.push_back(c);  b_.push_back(s);  c_.push_back(mn1);
    a_.push_back(-s); b_.push_back(c);  c_.push_back(mn2);
    a_.push_back(c);  b_.push_back(s);  c_.push_back(mx1);
    a_.push_back(-s); b_.push_back(c);  c_.push_back(mx2);

    return calc_rect_contour();
}

double LShapedFIT::calc_area_criterion(const Mat& c1, const Mat& c2) {
    int n = c1.rows;
    std::vector<double> v1(n), v2(n);
    for (int i = 0; i < n; i++) {
        v1[i] = c1.at(i,0);
        v2[i] = c2.at(i,0);
    }

    double mn1, mx1, mn2, mx2;
    minMax(v1, mn1, mx1);
    minMax(v2, mn2, mx2);

    return -(mx1 - mn1) * (mx2 - mn2);
}

double LShapedFIT::calc_nearest_criterion(const Mat& c1, const Mat& c2) {
    int n = c1.rows;
    std::vector<double> v1(n), v2(n);
    for (int i = 0; i < n; i++) {
        v1[i] = c1.at(i,0);
        v2[i] = c2.at(i,0);
    }

    double mn1, mx1, mn2, mx2;
    minMax(v1, mn1, mx1);
    minMax(v2, mn2, mx2);

    double sum = 0;
    for (int i = 0; i < n; i++) {
        double d1 = std::min(std::abs(mx1 - v1[i]), std::abs(v1[i] - mn1));
        double d2 = std::min(std::abs(mx2 - v2[i]), std::abs(v2[i] - mn2));
        double d  = std::max(std::min(d1, d2), min_dist_of_nearest_crit_);
        sum += 1.0 / d;
    }
    return sum;
}

double LShapedFIT::calc_variances_criterion(const Mat& c1, const Mat& c2) {
    int n = c1.rows;
    std::vector<double> v1(n), v2(n);
    for (int i = 0; i < n; i++) {
        v1[i] = c1.at(i,0);
        v2[i] = c2.at(i,0);
    }

    double mn1, mx1, mn2, mx2;
    minMax(v1, mn1, mx1);
    minMax(v2, mn2, mx2);

    std::vector<double> e1, e2;
    for (int i = 0; i < n; i++) {
        double d1 = std::min(std::abs(mx1 - v1[i]), std::abs(v1[i] - mn1));
        double d2 = std::min(std::abs(mx2 - v2[i]), std::abs(v2[i] - mn2));
        if (d1 < d2) e1.push_back(d1);
        else         e2.push_back(d2);
    }

    double vA = e1.empty() ? 0 : -calc_var(e1);
    double vB = e2.empty() ? 0 : -calc_var(e2);
    return vA + vB;
}

double LShapedFIT::calc_var(const std::vector<double>& v) {
    double mean = std::accumulate(v.begin(), v.end(), 0.0) / v.size();
    double acc = 0;
    for (double d : v) acc += (d - mean)*(d - mean);
    return acc / (v.size() - 1);    // true variance
}

void LShapedFIT::calc_cross_point(double a0, double a1,
                                  double b0, double b1,
                                  double c0, double c1,
                                  double& x, double& y)
{
    double det = a0*b1 - a1*b0;
    if (std::abs(det) < 1e-12) {
        x = y = std::numeric_limits<double>::quiet_NaN();
        return;
    }
    x = (b0*(-c1) - b1*(-c0)) / det;
    y = (a1*(-c0) - a0*(-c1)) / det;
}

RotatedRect LShapedFIT::calc_rect_contour() {
    vertex_pts_.clear();
    double x,y;

    calc_cross_point(a_[0],a_[1], b_[0],b_[1], c_[0],c_[1], x,y);
    vertex_pts_.push_back(Point2f(x,y));

    calc_cross_point(a_[1],a_[2], b_[1],b_[2], c_[1],c_[2], x,y);
    vertex_pts_.push_back(Point2f(x,y));

    calc_cross_point(a_[2],a_[3], b_[2],b_[3], c_[2],c_[3], x,y);
    vertex_pts_.push_back(Point2f(x,y));

    calc_cross_point(a_[3],a_[0], b_[3],b_[0], c_[3],c_[0], x,y);
    vertex_pts_.push_back(Point2f(x,y));

    return build_rotated_rect_from_vertices(vertex_pts_);
}

RotatedRect LShapedFIT::build_rotated_rect_from_vertices(std::vector<Point2f>& pts)
{
    if (pts.size() != 4) return RotatedRect();

    // compute center
    double cx=0, cy=0;
    for (auto& p : pts) { cx += p.x; cy += p.y; }
    cx /= 4; cy /= 4;

    // sort CCW around center
    std::sort(pts.begin(), pts.end(),
             [&](const Point2f& A, const Point2f& B){
                 return std::atan2(A.y - cy, A.x - cx)
                      < std::atan2(B.y - cy, B.x - cx);
             });

    // compute edge lengths
    auto dist = [&](const Point2f& a, const Point2f& b){
        double dx = b.x - a.x, dy = b.y - a.y;
        return std::sqrt(dx*dx + dy*dy);
    };

    double w = dist(pts[0], pts[1]);
    double h = dist(pts[1], pts[2]);
    double angle = std::atan2(pts[1].y - pts[0].y,
                              pts[1].x - pts[0].x);

    // ensure width < height
    if (w > h) {
        std::swap(w, h);
        angle += M_PI/2;
    }

    return RotatedRect(Point2f(cx,cy), Point2f(w,h), angle);
}

std::vector<Point2f> LShapedFIT::getRectVertex() {
    return vertex_pts_;
}

} // namespace geo


/* 

- Provide a cleaned-up, modernized C++ version
(e.g., remove repeated code, replace cv::Mat scalar loops with vectorized ops, remove unnecessary heap allocs)

- Identify bugs or undefined behavior
Examples I already noticed:
- minimal_cost initialization is conceptually wrong for maximizing a cost function.
- sort() is unnecessary for many steps.
- Misalignment of how c1_deep and c2_deep are used vs. the meaning of original points.
- A few potential division-by-zero risks.
- Potential incorrect vertex ordering due to line intersection ordering.

- Optimize speed
This implementation uses slow operations (Mat × Mat in a loop, many allocations). I can show how to optimize by:
- Precompute point arrays
- Use Eigen
- Avoid repeated trig
- Use SIMD-friendly data layout

*/
