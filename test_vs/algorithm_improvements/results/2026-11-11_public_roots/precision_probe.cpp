#define main eigen_regression_main
#include "/opt/ModelingCodingAutomationProject/python_numpy_to_cpp/test_vs/algorithm_improvements/eigen_regression.cpp"
#undef main

int main() {
  double sqrt_error = 0;
  for (int i = 0; i <= 10000; ++i) {
    double x = 1.0 + 15.0 * double(i) / 10000;
    double error = std::abs(Base::Math::sqrt(x) / std::sqrt(x) - 1.0);
    if (error > sqrt_error) { sqrt_error = error; }
  }
  std::printf("sqrt max relative error [1,16]=%.17g\n", sqrt_error);
  std::printf("sqrt2 relative error=%.17g\n", Base::Math::sqrt(2.0) / std::sqrt(2.0) - 1.0);
  std::printf("tiny root relative error=%.17g\n", EigenOperation::square_root(1e-300) / 1e-150 - 1.0);
  Matrix<double,3,3> a({{4,1,0},{1,3,1},{0,1,2}});
  Vector<double,3> truth({1,-2,3});
  GmresSolver<double,3> gmres;
  gmres.relative_tolerance=1e-12;gmres.division_min=1e-20;
  Vector<double,3> solution=gmres.solve(a,a*truth,Vector<double,3>());
  double square=0;
  for(int i=0;i<3;++i){square+=(solution[i]-truth[i])*(solution[i]-truth[i]);}
  std::printf("gmres3 truth error=%.17g residual=%.17g converged=%d\n",std::sqrt(square),gmres.get_residual_norm(),int(gmres.is_converged()));
  Matrix<double,6,6> nonsymmetric=rotated_diagonal<double,6>(true);
  EigenSolverReal<double,6> real;
  real.iteration_max=100;real.solve_eigen_values_and_vectors(nonsymmetric);
  double eigen_error=0;
  for(auto x:real.get_eigen_values()){
    double nearest=100;
    for(int j=1;j<=6;++j){double e=std::abs(x-double(j));if(e<nearest){nearest=e;}}
    if(nearest>eigen_error){eigen_error=nearest;}
  }
  std::printf("eigen6 spectrum max error=%.17g residual=%.17g converged=%d\n",eigen_error,real.get_eigenvector_residual(),int(real.eigen_values_converged()));
  Matrix<double,6,6> symmetric=rotated_diagonal<double,6>();
  EigenSolverSymmetric<double,6> jacobi;jacobi.iteration_max=10;jacobi.solve_eigen_values_and_vectors(symmetric);
  double jacobi_error=0,orthogonal_error=0,residual_error=0;
  auto values=jacobi.get_eigen_values();auto vectors=jacobi.get_eigen_vectors();
  for(auto x:values){double nearest=100;for(int j=1;j<=6;++j){double e=std::abs(x-double(j));if(e<nearest){nearest=e;}}if(nearest>jacobi_error){jacobi_error=nearest;}}
  for(int i=0;i<6;++i){for(int j=0;j<6;++j){double dot=0,residual=-vectors(i,j)*values[j];for(int k=0;k<6;++k){dot+=vectors(k,i)*vectors(k,j);residual+=symmetric(i,k)*vectors(k,j);}double ortho=std::abs(dot-double(i==j));if(ortho>orthogonal_error){orthogonal_error=ortho;}if(std::abs(residual)>residual_error){residual_error=std::abs(residual);}}}
  std::printf("jacobi6 spectrum max error=%.17g orthogonality=%.17g residual=%.17g\n",jacobi_error,orthogonal_error,residual_error);
}
