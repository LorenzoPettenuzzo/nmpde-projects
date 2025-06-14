#ifndef FISHERKOLMOGOROV_HPP
#define FISHERKOLMOGOROV_HPP

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/quadrature_lib.h>

#include <deal.II/distributed/fully_distributed_tria.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_simplex_p.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/fe_values_extractors.h>
#include <deal.II/fe/mapping_fe.h>

#include <deal.II/grid/grid_in.h>

#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/trilinos_precondition.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/grid/grid_tools.h>

#include <fstream>
#include <iostream>

using namespace dealii;

// Class representing the non-linear diffusion problem.
template <int dim>
class FisherKolmogorov
{
  public:

  class AnisotropicDiffusion : public Function<dim> {
      public:
      virtual double
      value(const Point<dim> & /*p*/,
            const unsigned int /*component*/ = 0) const override
      {
          return 1.0;
      }
  };

  class IsotropicDiffusion : public Function<dim> {
    public:
    virtual double
    value(const Point<dim> & /*p*/,
          const unsigned int /*component*/ = 0) const override
    {
        return 150.0;   // 150 mm^2/year  <-  1.5 cm^2/year
    }
};
    
  // Function for the alpha coefficient.
  class FunctionAlpha : public Function<dim>{
      public:
      virtual double
      value(const Point<dim> & /*p*/,
            const unsigned int /*component*/ = 0) const override
      {
        return 0.5;   //    α = 0.6/year in white and α = 0.3/year in gray matter
      }
  };

  // Function for initial conditions.
  class FunctionU0 : public Function<dim>
  {
  public:
    virtual double
    value(const Point<dim> & p,
          const unsigned int /*component*/ = 0) const override
    {
      if (p[2]<15)    //continuous initial seeding
        return 0.1;
      else if (p[2]>=15 && p[2]<20)
          return (-0.02 * p[2] + 0.4);
      else
        return 0.0;
    }
  };

  class CritTime0 : public Function<dim>
  {
    public:
    virtual double
    value(const Point<dim> & /*p*/,
          const unsigned int /*component*/ = 0) const override
    {
        return 0.0;
    }
  };


  // Constructor. We provide the final time, time step Delta t and theta method
  // parameter as constructor arguments.
  FisherKolmogorov (const std::string  &mesh_file_name_,
                    const unsigned int &r_,
                    const double       &T_,
                    const double       &deltat_)
    : mpi_size(Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD))
    , mpi_rank(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD))
    , pcout(std::cout, mpi_rank == 0)
    , T(T_)
    , mesh_file_name(mesh_file_name_)
    , r(r_)
    , deltat(deltat_)
    , mesh(MPI_COMM_WORLD)
  {}

  // Initialization.
  void
  setup();

  // Solve the problem.
  void
  solve();

  protected:

  // Assemble the tangent problem.
  void
  assemble_system();

  // Solve the linear system associated to the tangent problem.
  void
  solve_linear_system();

  // Solve the problem for one time step using Newton's method.
  void
  solve_newton();

  // Output.
  void
  output(const unsigned int &time_step) const;

  // MPI parallel. /////////////////////////////////////////////////////////////

  // Number of MPI processes.
  const unsigned int mpi_size;

  // This MPI process.
  const unsigned int mpi_rank;

  // Parallel output stream.
  ConditionalOStream pcout;

  // Problem definition. ///////////////////////////////////////////////////////

  // Diffusion coefficients
  AnisotropicDiffusion anisotropic_diff;

  IsotropicDiffusion isotropic_diff;

  // Alpha coefficient.
  FunctionAlpha alpha;

  // Initial conditions.
  FunctionU0 c_0;

  // Initial conditions for the critical time.
  CritTime0 c_crit;

  // Current time.
  double time;

  // Final time.
  const double T;

  // Discretization. ///////////////////////////////////////////////////////////

  // Mesh file name.
  const std::string mesh_file_name;

  // Polynomial degree.
  const unsigned int r;

  // Time step.
  const double deltat;

  // Mesh.
  parallel::fullydistributed::Triangulation<dim> mesh;

  // Finite element space.
  std::unique_ptr<FiniteElement<dim>> fe;

  // Quadrature formula.
  std::unique_ptr<Quadrature<dim>> quadrature;

  // DoF handler.
  DoFHandler<dim> dof_handler;

  // DoFs owned by current process.
  IndexSet locally_owned_dofs;

  // DoFs relevant to the current process (including ghost DoFs).
  IndexSet locally_relevant_dofs;

  // Jacobian matrix.
  TrilinosWrappers::SparseMatrix jacobian_matrix;

  // Residual vector.
  TrilinosWrappers::MPI::Vector residual_vector;

  // Increment of the solution between Newton iterations.
  TrilinosWrappers::MPI::Vector delta_owned;

  // System solution (without ghost elements).
  TrilinosWrappers::MPI::Vector solution_owned;

  // System solution (including ghost elements).
  TrilinosWrappers::MPI::Vector solution;

  // System solution at previous time step.
  TrilinosWrappers::MPI::Vector solution_old;

  // System solution for the critical time step.
  TrilinosWrappers::MPI::Vector critical_time_solution;
};

#include "FisherKolmogorov.cpp"

#endif // FISHERKOLMOGOROV_HPP
