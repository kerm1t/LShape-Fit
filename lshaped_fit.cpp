#include "lshaped_fit.h"
#include <iostream>

/*
PROBLEM CONTEXT
    Given a cloud of 2D points (e.g., LiDAR points around a vehicle), we want to fit the best-aligned rectangle around them.
    But:
    - The object may be rotated.
    - he point cloud may form an L-shape (like a car corner).
    - We want the rectangle that best follows the “structure” of the data.
    So we try many orientations and choose the best one.

ALGORITHM OVERVIEW
    The algorithm rotates a coordinate system, measures how tightly the points form an L-shape at each angle,
    and chooses the angle that gives the tightest rectangle around the projected points.
*/
#define M_PI 3.14159265358979323846

 namespace geo {
LShapedFIT::LShapedFIT() {
    min_dist_of_nearest_crit_ = 0.01;
    dtheta_deg_for_search_    = 1.0;
    criterion_ = LShapedFIT::VARIANCE;
}

LShapedFIT::~LShapedFIT() {}

void LShapedFIT::minMax(const std::vector<double>& v, double& mn, double& mx) {
    mn = std::numeric_limits<double>::max();
    mx = -mn;
    for (double d : v) {
        if (d < mn) mn = d;
        if (d > mx) mx = d;
    }
}

/* ALGORITHM STEPS
    1. Try many possible angles
        We don’t know the object’s orientation. So we search angles from 0° → 90° in small steps (e.g., 1°).
        For each angle θ, we imagine rotating two axes:
        - e1 = the x-axis rotated by θ
        - e2 = a perpendicular axis
        They form a rotated coordinate system.

    2. Project all the points onto the two axes

    3. Use a criterion to measure “how good” this angle is

    4. Pick the angle with the best score
        After looping all angles:
        The best angle = strongest L-shape alignment
        This gives us 4 bounding lines:
        two parallel to e1
        two parallel to e2
        These form the rectangle.

    5. Compute rectangle corners

    6. Build the final RotatedRect
        Given the 4 corners:
        - center = average of corners
        - width = distance between corner 1 → 2
        - height = distance between corner 2 → 3
        - angle = orientation of one side
*/
RotatedRect LShapedFIT::FitBox(std::vector<Point2f>* pointcloud_ptr)
{
    auto& points = *pointcloud_ptr;
    if (points.size() < 3) return RotatedRect();

    // Build matrix
    Mat Matrix_pts(points.size(), 2, 0.0);
    for (size_t i = 0; i < points.size(); ++i) {
        Matrix_pts.at(i,0) = points[i].x;
        Matrix_pts.at(i,1) = points[i].y;
    }

    double dtheta = dtheta_deg_for_search_ * M_PI / 180.0;
    double best_cost = -std::numeric_limits<double>::max();
    double best_theta = -1;

    int loop_number = std::ceil((M_PI / 2 - dtheta) / dtheta);

    for (int k = 0; k < loop_number; ++k) {
        double theta = k * dtheta;
        if (theta >= (M_PI/2 - dtheta)) break;

        double c = std::cos(theta);
        double s = std::sin(theta);

        // Basis vectors
        double e1x = c, e1y = s;
        double e2x = -s, e2y = c;

        // c1 = projection on e1, c2 on e2
        Mat c1(points.size(), 1);
        Mat c2(points.size(), 1);

        for (size_t i = 0; i < points.size(); i++) {
            double x = Matrix_pts.at(i,0);
            double y = Matrix_pts.at(i,1);
            c1.at(i,0) = x*e1x + y*e1y;
            c2.at(i,0) = x*e2x + y*e2y;
        }

        double cost = 0;
        if (criterion_ == AREA)
            cost = calc_area_criterion(c1, c2);
        else if (criterion_ == NEAREST)
            cost = calc_nearest_criterion(c1, c2);
        else
            cost = calc_variances_criterion(c1, c2);

        if (cost > best_cost) {
            best_cost = cost;
            best_theta = theta;
        }
    }

    if (best_theta < 0) return RotatedRect();

    double c = std::cos(best_theta);
    double s = std::sin(best_theta);

    // Final projections
    Mat c1(points.size(), 1);
    Mat c2(points.size(), 1);

    for (size_t i = 0; i < points.size(); i++) {
        double x = Matrix_pts.at(i,0);
        double y = Matrix_pts.at(i,1);
        c1.at(i,0) = x*c + y*s;
        c2.at(i,0) = -x*s + y*c;
    }

    // find min/max
    std::vector<double> v1(points.size());
    std::vector<double> v2(points.size());
    for (size_t i = 0; i < points.size(); i++) {
        v1[i] = c1.at(i,0);
        v2[i] = c2.at(i,0);
    }

    double min1, max1, min2, max2;
    minMax(v1, min1, max1);
    minMax(v2, min2, max2);

    a_.clear(); b_.clear(); c_.clear();
    a_.push_back(c);  b_.push_back(s);  c_.push_back(min1);
    a_.push_back(-s); b_.push_back(c);  c_.push_back(min2);
    a_.push_back(c);  b_.push_back(s);  c_.push_back(max1);
    a_.push_back(-s); b_.push_back(c);  c_.push_back(max2);

    return calc_rect_contour();
}

// --------------------------
// Criterion functions
// --------------------------

// Criterion A: AREA
double LShapedFIT::calc_area_criterion(const Mat& c1, const Mat& c2) {
    std::vector<double> v1(c1.rows), v2(c1.rows);
    for (int i = 0; i < c1.rows; i++) {
        v1[i] = c1.at(i,0);
        v2[i] = c2.at(i,0);
    }
    double mn1, mx1, mn2, mx2;
    minMax(v1, mn1, mx1);
    minMax(v2, mn2, mx2);
    return - (mx1 - mn1) * (mx2 - mn2);
}

/* Criterion B: NEAREST
For each point:
- Find its distance to the closest rectangle border
- Reward rectangle orientations where points are close to their sides
- Penalize orientations that put many points far from edges
*/
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

// Criterion C: VARIANCE
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

    std::vector<double> d1(n), d2(n);
    for (int i = 0; i < n; i++) {
        d1[i] = std::min(std::abs(mx1 - v1[i]), std::abs(v1[i] - mn1));
        d2[i] = std::min(std::abs(mx2 - v2[i]), std::abs(v2[i] - mn2));
    }

    std::vector<double> e1, e2;
    for (int i = 0; i < n; i++) {
        if (d1[i] < d2[i]) e1.push_back(d1[i]);
        else               e2.push_back(d2[i]);
    }

    double vA = e1.empty() ? 0 : -calc_var(e1);
    double vB = e2.empty() ? 0 : -calc_var(e2);
    return vA + vB;
}

double LShapedFIT::calc_var(const std::vector<double>& v) {
    double mean = std::accumulate(v.begin(), v.end(), 0.0) / v.size();
    double acc = 0;
    for (double d : v)
        acc += (d - mean) * (d - mean);
    return std::sqrt(acc / (v.size() - 1));
}

// --------------------------
// Geometry
// --------------------------

void LShapedFIT::calc_cross_point(const double a0, const double a1,
                                  const double b0, const double b1,
                                  const double c0, const double c1,
                                  double& x, double& y)
{
    double det = a0 * b1 - a1 * b0;
    x = (b0 * (-c1) - b1 * (-c0)) / det;
    y = (a1 * (-c0) - a0 * (-c1)) / det;
}

RotatedRect LShapedFIT::calc_rect_contour() {
    vertex_pts_.clear();

    double x, y;

    calc_cross_point(a_[0], a_[1], b_[0], b_[1], c_[0], c_[1], x, y);
    vertex_pts_.push_back(Point2f(x, y));

    calc_cross_point(a_[1], a_[2], b_[1], b_[2], c_[1], c_[2], x, y);
    vertex_pts_.push_back(Point2f(x, y));

    calc_cross_point(a_[2], a_[3], b_[2], b_[3], c_[2], c_[3], x, y);
    vertex_pts_.push_back(Point2f(x, y));

    calc_cross_point(a_[3], a_[0], b_[3], b_[0], c_[3], c_[0], x, y);
    vertex_pts_.push_back(Point2f(x, y));

    return build_rotated_rect_from_vertices(vertex_pts_);
}

RotatedRect LShapedFIT::build_rotated_rect_from_vertices(const std::vector<Point2f>& pts) {
    if (pts.size() != 4) return RotatedRect();

    // Compute center
    double cx = 0, cy = 0;
    for (auto& p : pts) { cx += p.x; cy += p.y; }
    cx /= 4; cy /= 4;

    // width = distance pts[0] to pts[1]
    double dx1 = pts[1].x - pts[0].x;
    double dy1 = pts[1].y - pts[0].y;
    double width = std::sqrt(dx1*dx1 + dy1*dy1);

    // height = distance pts[1] to pts[2]
    double dx2 = pts[2].x - pts[1].x;
    double dy2 = pts[2].y - pts[1].y;
    double height = std::sqrt(dx2*dx2 + dy2*dy2);

    double angle = std::atan2(dy1, dx1);

    return RotatedRect(Point2f(cx, cy), Point2f(width, height), angle);
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
