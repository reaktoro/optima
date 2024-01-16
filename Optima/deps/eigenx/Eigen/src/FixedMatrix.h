// This file is part of Eigenx, an extension of Eigen.
//
// Copyright © 2018-2024 Allan Leal
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_FIXEDMATRIX_H
#define EIGEN_FIXEDMATRIX_H

namespace Eigen {

/// Used to deal with matrices and vectors that cannot be resized after construction.
template<typename Scalar, int Rows, int Cols, int Options, int MaxRows, int MaxCols>
class FixedMatrix : public Matrix<Scalar, Rows, Cols, Options, MaxRows, MaxCols>
{
public:
    using Base = Matrix<Scalar, Rows, Cols, Options, MaxRows, MaxCols>;
    using Base::Base;

    void resize(Index rows, Index cols) = delete;
    void resize(Index size) = delete;
    void resize(NoChange_t, Index cols) = delete;
    void resize(Index rows, NoChange_t) = delete;
    void conservativeResize(Index rows, Index cols) = delete;
    void conservativeResize(Index rows, NoChange_t) = delete;
    void conservativeResize(NoChange_t, Index cols) = delete;
    void conservativeResize(Index size) = delete;
    template<typename OtherDerived> void resizeLike(const EigenBase<OtherDerived>& other) = delete;
    template<typename OtherDerived> void conservativeResizeLike(const DenseBase<OtherDerived>& other) = delete;

    void __resize(Index rows, Index cols) { Base::resize(rows, cols); }
    void __resize(Index size) { Base::resize(size); }
    void __resize(NoChange_t, Index cols) { Base::resize(NoChange, cols); }
    void __resize(Index rows, NoChange_t) { Base::resize(rows, NoChange); }
    void __conservativeResize(Index rows, Index cols) { Base::conservativeResize(rows, cols); }
    void __conservativeResize(Index rows, NoChange_t) { Base::conservativeResize(rows, NoChange); }
    void __conservativeResize(NoChange_t, Index cols) { Base::conservativeResize(NoChange, cols); }
    void __conservativeResize(Index size) { Base::conservativeResize(size); }
    template<typename OtherDerived> void __resizeLike(const EigenBase<OtherDerived>& other) { Base::resizeLike(other); }
    template<typename OtherDerived> void __conservativeResizeLike(const DenseBase<OtherDerived>& other) { Base::conservativeResizeLike(other); }
    template<typename OtherDerived> void __assign(const MatrixBase<OtherDerived>& other) { Base::operator=(other); }

    template<typename OtherDerived>
    FixedMatrix& operator=(const MatrixBase<OtherDerived>& other)
    {
        if(this->size() == 0 && other.size() == 0) // avoid runtime error below when dimensions differ, even though one of them is zero
            return *this;
        if(this->rows() != other.rows() || this->cols() != other.cols())
            throw std::runtime_error("\033[1;31m***ERROR*** Cannot implicitly resize a matrix/vector with fixed dimensions.\033[0m");
        Base::operator=(other);
        return *this;
    }
};

} // namespace Eigen

#endif // EIGEN_FIXEDMATRIX_H
