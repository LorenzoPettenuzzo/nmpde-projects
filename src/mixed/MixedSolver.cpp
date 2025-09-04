#ifndef LINEARFISHERKOLMOGOROV_CPP
#define LINEARFISHERKOLMOGOROV_CPP
#include "MixedSolver.hpp"

template <int dim>
void
MixedSolver<dim>::setup()
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
    DoFTools::extract_locally_relevant_dofs(dof_handler, locally_relevant_dofs);

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
    lhs_matrix.reinit(sparsity);

    pcout << "  Initializing the system right-hand side" << std::endl;
    system_rhs.reinit(locally_owned_dofs, MPI_COMM_WORLD);
    pcout << "  Initializing the solution vector" << std::endl;
    solution_owned.reinit(locally_owned_dofs, MPI_COMM_WORLD);

    solution.reinit(locally_owned_dofs, locally_relevant_dofs, MPI_COMM_WORLD);
    solution_old = solution;

    critical_time_solution.reinit(locally_owned_dofs, MPI_COMM_WORLD);
  }
}

template <int dim>
void
MixedSolver<dim>::assemble_lhs_matrix()
{
  pcout << "===============================================" << std::endl;
  pcout << "Assembling the system matrices" << std::endl;

  const unsigned int dofs_per_cell = fe->dofs_per_cell;
  const unsigned int n_q           = quadrature->size();

  FEValues<dim> fe_values(*fe,
                          *quadrature,
                          update_values | update_gradients |
                            update_quadrature_points | update_JxW_values );

  FullMatrix<double> cell_lhs_matrix(dofs_per_cell, dofs_per_cell);

  std::vector<types::global_dof_index> dof_indices(dofs_per_cell);

  lhs_matrix = 0.0;

  std::vector<double>         solution_loc(n_q);

  for (const auto &cell : dof_handler.active_cell_iterators())
    {
      if (!cell->is_locally_owned())
        continue;

      fe_values.reinit(cell);

      cell_lhs_matrix = 0.0;

      fe_values.get_function_values(solution, solution_loc);
      
      for (unsigned int q = 0; q < n_q; ++q)
        {
          // Evaluate coefficients on this quadrature node.
          const double alpha_loc = alpha.value(fe_values.quadrature_point(q));

          for (unsigned int i = 0; i < dofs_per_cell; ++i)
            {
              for (unsigned int j = 0; j < dofs_per_cell; ++j)
                {
                  // 1/deltat * integral(psi_j * psi_i)
                  cell_lhs_matrix(i,j) += (1.0/deltat) *
                    fe_values.shape_value(i,q) *
                    fe_values.shape_value(j,q) *
                    fe_values.JxW(q);

                  // d^ext * integral(grad(psi_j)*grad(psi_i))
                  cell_lhs_matrix(i,j) += isotropic_diff *
                    fe_values.shape_grad(i,q) *
                    fe_values.shape_grad(j,q) *
                    fe_values.JxW(q);

                  // - alpha * integral(psi_j*psi_i)
                  cell_lhs_matrix(i,j) += (-1) * alpha_loc *
                    fe_values.shape_value(i,q) *
                    fe_values.shape_value(j,q) *
                    fe_values.JxW(q);
                }
            }
        }

      cell->get_dof_indices(dof_indices);

      lhs_matrix.add(dof_indices, cell_lhs_matrix);
    }

  lhs_matrix.compress(VectorOperation::add);

}

template <int dim>
void
MixedSolver<dim>::assemble_rhs(const double & /*time*/)
{
  const unsigned int dofs_per_cell = fe->dofs_per_cell;
  const unsigned int n_q           = quadrature->size();

  FEValues<dim> fe_values(*fe,
                          *quadrature,
                          update_values | update_quadrature_points |
                            update_JxW_values);

  Vector<double> cell_rhs(dofs_per_cell);

  std::vector<types::global_dof_index> dof_indices(dofs_per_cell);

  std::vector<double> solution_old_loc(n_q);

  system_rhs = 0.0;

  for (const auto &cell : dof_handler.active_cell_iterators())
    {
      if (!cell->is_locally_owned())
        continue;

      fe_values.reinit(cell);

      fe_values.get_function_values(solution_old, solution_old_loc);

      cell_rhs = 0.0;

      for (unsigned int q = 0; q < n_q; ++q)
        {

          const double alpha_loc = alpha.value(fe_values.quadrature_point(q));

          for (unsigned int i = 0; i < dofs_per_cell; ++i)
            {

              cell_rhs(i) += (1.0/deltat) *
                solution_old_loc[q] *
                fe_values.shape_value(i,q) *
                fe_values.JxW(q);

              cell_rhs(i) += (-1) * alpha_loc *
                solution_old_loc[q] * 
                solution_old_loc[q] *
                fe_values.shape_value(i,q) *
                fe_values.JxW(q);

            }
        }

      cell->get_dof_indices(dof_indices);
      system_rhs.add(dof_indices, cell_rhs);
    }

  system_rhs.compress(VectorOperation::add);

}

template <int dim>
void
MixedSolver<dim>::solve_time_step()
{
  SolverControl solver_control(1000, 1e-6 * system_rhs.l2_norm());

  SolverBicgstab<TrilinosWrappers::MPI::Vector> solver(solver_control);
  TrilinosWrappers::PreconditionSSOR      preconditioner;
  preconditioner.initialize(
    lhs_matrix, TrilinosWrappers::PreconditionSSOR::AdditionalData(1.0));

  solver.solve(lhs_matrix, solution_owned, system_rhs, preconditioner);
  pcout << "  " << solver_control.last_step() << " CG iterations" << std::endl;

  solution = solution_owned;
}

template <int dim>
void
MixedSolver<dim>::output(const unsigned int &time_step) const
{
  DataOut<dim> data_out;
  data_out.add_data_vector(dof_handler, solution, "u");

  std::vector<unsigned int> partition_int(mesh.n_active_cells());
  GridTools::get_subdomain_association(mesh, partition_int);
  const Vector<double> partitioning(partition_int.begin(), partition_int.end());
  data_out.add_data_vector(partitioning, "partitioning");

  data_out.build_patches();

  data_out.write_vtu_with_pvtu_record(
    "./", "output", time_step, MPI_COMM_WORLD, 3);
}

template <int dim>
void
MixedSolver<dim>::solve()
{

  pcout << "===============================================" << std::endl;

  time = 0.0;

  // Apply the initial condition.
  {
    pcout << "Applying the initial condition" << std::endl;

    //exact_solution.set_time(time);
    VectorTools::interpolate(dof_handler, c_0, solution_owned);
    solution = solution_owned;

    // initialize the critical time solution
    VectorTools::interpolate(dof_handler, c_crit, critical_time_solution);

    // Output the initial solution. [OPTIONAL]
    //output(0);

    pcout << "-----------------------------------------------" << std::endl;
  }

  std::ofstream output_file("critical_fraction.txt");

  unsigned int time_step = 0;

  double crit_frac = 0.0;
  double crit_frac_global = 0.0;

  while (time < T - 0.5 * deltat)
    {
      time += deltat;
      ++time_step;

      crit_frac = 0.0;
      crit_frac_global = 0.0;

      pcout << "n = " << std::setw(3) << time_step << ", t = " << std::setw(5)
            << time << ":" << std::flush;

      solution_old = solution;

      assemble_lhs_matrix();
      assemble_rhs(time);
      solve_time_step();

      //Output current solution. [OPTIONAL]
      //output(time_step);

      // Check for time of critical concentration.
      std::vector<types::global_dof_index> dof_indices;
      for (const auto &cell : dof_handler.active_cell_iterators())
        {
          if (!cell->is_locally_owned())
            continue;

          {
            dof_indices.resize(cell->get_fe().dofs_per_cell);
            cell->get_dof_indices(dof_indices);
            for (unsigned int i = 0; i < dof_indices.size(); ++i)
              {
                if ((solution[dof_indices[i]] > 0.95))
                  {
                  crit_frac += 1.0;
                  if (critical_time_solution[dof_indices[i]] == 0.0)
                    {
                    // Store the critical time solution.
                    critical_time_solution[dof_indices[i]] = time;
                    }
                  }
              }
          }
        }
      critical_time_solution.compress(VectorOperation::insert);

      //Update critical fraction file.
      crit_frac_global = Utilities::MPI::sum(crit_frac, MPI_COMM_WORLD);
      //Process 0 writes the critical fraction to the file.
      if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
        {
          output_file << crit_frac_global
                      << std::endl;
          pcout << "Critical fraction = " << crit_frac_global
                << std::endl;
        }
      //output_file << crit_frac_global / dof_handler.n_dofs() << std::endl;
      //output_file.flush();

      pcout << std::endl;
    }

    //Output critical time solution.
    DataOut<dim> data_out;
    data_out.add_data_vector(dof_handler, critical_time_solution,
                              "critical_time");
    data_out.build_patches();
    data_out.write_vtu_with_pvtu_record(
      "./", "output_crit_time", time_step, MPI_COMM_WORLD, 3);

    //Close the output file.
    output_file.close();
}

#endif