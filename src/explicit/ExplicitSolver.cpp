#ifndef EXPLICIT_SOLVER_CPP
#define EXPLICIT_SOLVER_CPP

#include "ExplicitSolver.hpp"

template<int dim>
void
ExplicitSolver<dim>::setup()
{
  // Create the mesh.
  {
    pcout << "Initializing the mesh" << std::endl;

    Triangulation<dim> mesh_serial;

    GridIn<dim> grid_in;
    grid_in.attach_triangulation(mesh_serial);

    std::ifstream grid_in_file(mesh_file_name);
    grid_in.read_msh(grid_in_file);

    GridTools::partition_triangulation(mpi_size, mesh_serial);
    const auto construction_data = TriangulationDescription::Utilities::
      create_description_from_triangulation(mesh_serial, MPI_COMM_WORLD);
    mesh.create_triangulation(construction_data);

    pcout << "  Number of elements = " << mesh.n_global_active_cells()
          << std::endl;
  }

  pcout << "-----------------------------------------------" << std::endl;

  // Initialize the finite element space.
  {
    pcout << "Initializing the finite element space" << std::endl;

    fe = std::make_unique<FE_SimplexP<dim>>(r);

    pcout << "  Degree                     = " << fe->degree << std::endl;
    pcout << "  DoFs per cell              = " << fe->dofs_per_cell
          << std::endl;

    quadrature = std::make_unique<QGaussSimplex<dim>>(r + 1);

    pcout << "  Quadrature points per cell = " << quadrature->size()
          << std::endl;
  }

  pcout << "-----------------------------------------------" << std::endl;

  // Initialize the DoF handler.
  {
    pcout << "Initializing the DoF handler" << std::endl;

    dof_handler.reinit(mesh);
    dof_handler.distribute_dofs(*fe);

    locally_owned_dofs = dof_handler.locally_owned_dofs();
    //locally_relevant_dofs =
      //DoFTools::extract_locally_relevant_dofs(dof_handler);
      DoFTools::extract_locally_relevant_dofs(dof_handler, locally_relevant_dofs);  // change signature of method

    pcout << "  Number of DoFs = " << dof_handler.n_dofs() << std::endl;
  }

  pcout << "-----------------------------------------------" << std::endl;

  // Initialize the linear system.
  {
    pcout << "Initializing the linear system" << std::endl;

    pcout << "  Initializing the sparsity pattern" << std::endl;

    TrilinosWrappers::SparsityPattern sparsity(locally_owned_dofs,
                                               MPI_COMM_WORLD);
    DoFTools::make_sparsity_pattern(dof_handler, sparsity);
    sparsity.compress();

    pcout << "  Initializing the matrices" << std::endl;
    mass_matrix.reinit(sparsity);
    stiffness_matrix.reinit(sparsity);
    lhs_matrix.reinit(sparsity);
    rhs_matrix.reinit(sparsity);

    pcout << "  Initializing the system right-hand side" << std::endl;
    system_rhs.reinit(locally_owned_dofs, MPI_COMM_WORLD);
    pcout << "  Initializing the solution vector" << std::endl;
    solution_owned.reinit(locally_owned_dofs, MPI_COMM_WORLD);
    solution.reinit(locally_owned_dofs, locally_relevant_dofs, MPI_COMM_WORLD);
  }
}

template<int dim>
void
ExplicitSolver<dim>::assemble_matrices()
{
  pcout << "===============================================" << std::endl;
  pcout << "Assembling the system matrices" << std::endl;

  const unsigned int dofs_per_cell = fe->dofs_per_cell;
  const unsigned int n_q           = quadrature->size();

  FEValues<dim> fe_values(*fe,
                          *quadrature,
                          update_values | update_gradients |
                            update_quadrature_points | update_JxW_values);

  FullMatrix<double> cell_mass_matrix(dofs_per_cell, dofs_per_cell);
  FullMatrix<double> cell_stiffness_matrix(dofs_per_cell, dofs_per_cell);

  std::vector<types::global_dof_index> dof_indices(dofs_per_cell);

  mass_matrix      = 0.0;
  stiffness_matrix = 0.0;

  // check h_min
  double min_h = std::numeric_limits<double>::max();

  for (const auto &cell : dof_handler.active_cell_iterators())
    {
      if (!cell->is_locally_owned())
        continue;

      // check h_min
      const double h_cell = cell->minimum_vertex_distance(); // for simplices: smallest edge length
      min_h  = std::min(min_h, h_cell);

      fe_values.reinit(cell);

      cell_mass_matrix      = 0.0;
      cell_stiffness_matrix = 0.0;

      for (unsigned int q = 0; q < n_q; ++q)
        {
          // Evaluate coefficients on this quadrature node.
          const double D_loc = D.value(fe_values.quadrature_point(q));

          for (unsigned int i = 0; i < dofs_per_cell; ++i)
            {
              for (unsigned int j = 0; j < dofs_per_cell; ++j)
                {
                  cell_mass_matrix(i, j) += fe_values.shape_value(i, q) *
                                            fe_values.shape_value(j, q) * 
                                            fe_values.JxW(q);

                  cell_stiffness_matrix(i, j) +=
                    D_loc * fe_values.shape_grad(i, q) *
                    fe_values.shape_grad(j, q) * fe_values.JxW(q);
                }
            }
        }

      cell->get_dof_indices(dof_indices);

      mass_matrix.add(dof_indices, cell_mass_matrix);
      stiffness_matrix.add(dof_indices, cell_stiffness_matrix);
    }

  mass_matrix.compress(VectorOperation::add);
  stiffness_matrix.compress(VectorOperation::add);

  // We build the matrix on the left-hand side of the algebraic problem (the one
  // that we'll invert at each timestep).
  lhs_matrix.copy_from(mass_matrix);

  // We build the matrix on the right-hand side (the one that multiplies the old
  // solution un).
  rhs_matrix.copy_from(mass_matrix);

  // check h_min
  min_h = Utilities::MPI::min(min_h, MPI_COMM_WORLD);
  pcout << "Minimum h: " << min_h << std::endl;
}

template<int dim>
void
ExplicitSolver<dim>::assemble_rhs(const double &time)
{
  const unsigned int dofs_per_cell = fe->dofs_per_cell;
  const unsigned int n_q           = quadrature->size();

  FEValues<dim> fe_values(*fe,
                          *quadrature,
                          update_values | update_quadrature_points |
                            update_JxW_values);

  Vector<double> cell_rhs(dofs_per_cell);

  std::vector<types::global_dof_index> dof_indices(dofs_per_cell);

  system_rhs = 0.0;

  alpha.set_time(time - deltat);

  IndexSet locally_relevant;
  DoFTools::extract_locally_relevant_dofs(dof_handler, locally_relevant);

  // Create a ghosted vector on the fly
  TrilinosWrappers::MPI::Vector solution_ghosted(solution_owned.locally_owned_elements(),
                                                locally_relevant,
                                                MPI_COMM_WORLD);


  solution_ghosted = solution_owned;
  std::vector<double> values;

  for (const auto &cell : dof_handler.active_cell_iterators())
    {
      if (!cell->is_locally_owned())
        continue;

      fe_values.reinit(cell);
      
      values.resize(n_q);
      
      fe_values.get_function_values(solution_ghosted, values);

      cell_rhs = 0.0;

      for (unsigned int q = 0; q < n_q; ++q)
        {
          // Compute the nonlinear term n(c^n, phi)
          const double alpha_loc = alpha.value(fe_values.quadrature_point(q));
          const double nonlinear = alpha_loc * values[q] * (1 - values[q]);
          for (unsigned int i = 0; i < dofs_per_cell; ++i)
            {
              cell_rhs(i) += deltat * nonlinear * fe_values.shape_value(i, q) * fe_values.JxW(q);
            }
        }

      cell->get_dof_indices(dof_indices);
      system_rhs.add(dof_indices, cell_rhs);
    }

  system_rhs.compress(VectorOperation::add);

  // Add the term that comes from the old solution.
  TrilinosWrappers::MPI::Vector Mu(locally_owned_dofs, MPI_COMM_WORLD);
  mass_matrix.vmult(Mu, solution_owned);
  system_rhs.add(1.0, Mu);

  TrilinosWrappers::MPI::Vector Ku(locally_owned_dofs, MPI_COMM_WORLD);
  stiffness_matrix.vmult(Ku, solution_owned);
  system_rhs.add(-deltat, Ku);
}

template<int dim>
void
ExplicitSolver<dim>::solve_time_step()
{
  SolverControl solver_control(1000, 1e-6 * system_rhs.l2_norm());

  SolverCG<TrilinosWrappers::MPI::Vector> solver(solver_control);
  TrilinosWrappers::PreconditionJacobi      preconditioner;
  preconditioner.initialize(
    lhs_matrix, TrilinosWrappers::PreconditionJacobi::AdditionalData(1.0));

  solver.solve(lhs_matrix, solution_owned, system_rhs, preconditioner);
  pcout << "  " << solver_control.last_step() << " CG iterations" << std::endl;

  solution = solution_owned;
}

template<int dim>
void
ExplicitSolver<dim>::output(const unsigned int &time_step) const
{
    DataOut<dim> data_out;
  data_out.add_data_vector(dof_handler, solution, "c");

  std::vector<unsigned int> partition_int(mesh.n_active_cells());
  GridTools::get_subdomain_association(mesh, partition_int);
  const Vector<double> partitioning(partition_int.begin(), partition_int.end());
  data_out.add_data_vector(partitioning, "partitioning");

  data_out.build_patches();

  data_out.write_vtu_with_pvtu_record(
    "./images/", "output", time_step, MPI_COMM_WORLD, 3);
}

template<int dim>
void
ExplicitSolver<dim>::solve()
{
  assemble_matrices();

  pcout << "===============================================" << std::endl;

  // Apply the initial condition.
  {
    pcout << "Applying the initial condition" << std::endl;
  
    VectorTools::interpolate(dof_handler, c_0, solution_owned);
    solution = solution_owned;

    // Output the initial solution.
    output(0);
    pcout << "-----------------------------------------------" << std::endl;
  }

  unsigned int time_step = 0;
  double       time      = 0;

  while (time < T)
    {
      time += deltat;
      ++time_step;

      pcout << "n = " << std::setw(3) << time_step << ", t = " << std::setw(5)
            << time << ":" << std::flush;

      assemble_rhs(time);
      solve_time_step();
      output(time_step);
    }
}

#endif