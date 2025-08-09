#ifndef LINEARFISHERKOLMOGOROV_HPP
#define LINEARFISHERKOLMOGOROV_HPP

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
class LinearFisherKolmogorov
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

  // Function for the mu coefficient.
  class FunctionAlpha : public Function<dim>
  {
  public:
    virtual double
    value(const Point<dim> & /*p*/,
          const unsigned int /*component*/ = 0) const override
    {
      return 0.5;   //    α = 0.6/year in white and α = 0.3/year in gray matter
    }
  };

  // Function for the forcing term.
  class ForcingTerm : public Function<dim>
  {
  public:
    virtual double
    value(const Point<dim> & /*p*/,
          const unsigned int /*component*/ = 0) const override
    {
        return 0.0;
    }
  };

  // Exact solution.
  class ExactSolution : public Function<dim>
  {
  public:
    virtual double
    value(const Point<dim> & /*p*/,
          const unsigned int /*component*/ = 0) const override
    {
        return 0.0;
    }

    virtual Tensor<1, dim>
    gradient(const Point<dim> & /*p*/,
             const unsigned int /*component*/ = 0) const override
    {

       Tensor<1, dim> result;

    //   // duex / dx
    //   result[0] = 2 * M_PI * std::sin(5 * M_PI * get_time()) *
    //               std::cos(2 * M_PI * p[0]) * std::sin(3 * M_PI * p[1]) *
    //               std::sin(4 * M_PI * p[2]);

    //   // duex / dy
    //   result[1] = 3 * M_PI * std::sin(5 * M_PI * get_time()) *
    //               std::sin(2 * M_PI * p[0]) * std::cos(3 * M_PI * p[1]) *
    //               std::sin(4 * M_PI * p[2]);

    //   // duex / dz
    //   result[2] = 4 * M_PI * std::sin(5 * M_PI * get_time()) *
    //               std::sin(2 * M_PI * p[0]) * std::sin(3 * M_PI * p[1]) *
    //               std::cos(4 * M_PI * p[2]);

       return result;
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
      if (p[2]<15)    //very raw starting seeding
        return 0.1;
      else if (p[2]>=15 && p[2]<20)
          return (-0.02 * p[2] + 0.4);
      else
        return 0.0;
    }
  };

  // Constructor. We provide the final time, time step Delta t 
  // parameter as constructor arguments.
  LinearFisherKolmogorov(const std::string  &mesh_file_name_,
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

  // Compute the error.
  double
  compute_error(const VectorTools::NormType &norm_type);

protected:
  // Assemble the mass and stiffness matrices.
  void
  assemble_lhs_matrix();

  // Assemble the right-hand side vector and the right-hand side matrix
  void
  assemble_rhs(const double &time);

  // Solve the problem for one time step.
  void
  solve_time_step();

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

  AnisotropicDiffusion anisotropic_diff;

  double isotropic_diff = 150.0; //  150 mm^2/year  <-  1.5 cm^2/year
                                 //  maybe d = 225 = 150*3/2 is better, moving from 2d to 3d

  // mu coefficient.
  FunctionAlpha alpha;

  // Forcing term.
  ForcingTerm forcing_term;

  // Exact solution.
  ExactSolution exact_solution;

  // Initial conditions.
  FunctionU0 c_0;

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

  // Mass matrix M / deltat.
  TrilinosWrappers::SparseMatrix lhs_matrix;

  // Stiffness matrix A.
  TrilinosWrappers::SparseMatrix rhs_matrix;

  // Right-hand side vector in the linear system.
  TrilinosWrappers::MPI::Vector system_rhs;

  // System solution (without ghost elements).
  TrilinosWrappers::MPI::Vector solution_owned;

  // System solution (including ghost elements).
  TrilinosWrappers::MPI::Vector solution;
};

#include "MixedSolver.cpp"

#endif