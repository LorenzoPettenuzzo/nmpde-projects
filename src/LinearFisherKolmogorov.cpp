#include "LinearFisherKolmogorov.hpp"

template <int dim>
void
LinearFisherKolmogorov<dim>::setup()
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
    rhs_matrix.reinit(sparsity);

    pcout << "  Initializing the system right-hand side" << std::endl;
    system_rhs.reinit(locally_owned_dofs, MPI_COMM_WORLD);
    pcout << "  Initializing the solution vector" << std::endl;
    solution_owned.reinit(locally_owned_dofs, MPI_COMM_WORLD);
    solution.reinit(locally_owned_dofs, locally_relevant_dofs, MPI_COMM_WORLD);
  }
}

template <int dim>
void
LinearFisherKolmogorov<dim>::assemble_lhs_matrix()
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

  for (const auto &cell : dof_handler.active_cell_iterators())
    {
      if (!cell->is_locally_owned())
        continue;

      fe_values.reinit(cell);

      cell_lhs_matrix = 0.0;

      for (unsigned int q = 0; q < n_q; ++q)
        {
          // Evaluate coefficients on this quadrature node.
          const double alpha_loc = alpha.value(fe_values.quadrature_point(q));

          for (unsigned int i = 0; i < dofs_per_cell; ++i)
            {
              for (unsigned int j = 0; j < dofs_per_cell; ++j)
                {
                  // 1/deltat * integral(psi_j * psi_i)
                  lhs_matrix(i,j) += (1.0/deltat) *
                    fe_values->shape_value(i,q) *
                    fe_values->shape_value(j,q) *
                    fe_values->JxW(q);

                  // alpha * sum(k=0...dofs_per_cell: integral(psi_k * psi_j * psi_i))
                  for (unsigned int k = 0; k < dofs_per_cell; ++k) {
                    lhs_matrix(i,j) += alpha_loc * 
                      fe_values->shape_value(k,q) *
                      fe_values->shape_value(i,q) *
                      fe_values->shape_value(j,q) *
                      fe_values->JxW(q);
                  }
                  
                  // d^ext * integral(grad(psi_j)*grad(psi_i))
                  lhs_matrix(i,j) += isotropic_diff *
                    fe_values->shape_grad(i,q) *
                    fe_values->shape_grad(j,q) *
                    fe_values->JxW(q);
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
LinearFisherKolmogorov<dim>::assemble_rhs(const double &time)
{
  const unsigned int dofs_per_cell = fe->dofs_per_cell;
  const unsigned int n_q           = quadrature->size();

  FEValues<dim> fe_values(*fe,
                          *quadrature,
                          update_values | update_quadrature_points |
                            update_JxW_values);

  Vector<double> cell_rhs(dofs_per_cell);
  FullMatrix<double> cell_rhs_matrix(dofs_per_cell, dofs_per_cell);

  std::vector<types::global_dof_index> dof_indices(dofs_per_cell);

  system_rhs = 0.0;

  for (const auto &cell : dof_handler.active_cell_iterators())
    {
      if (!cell->is_locally_owned())
        continue;

      fe_values.reinit(cell);

      //cell_rhs = 0.0;
      cell_rhs_matrix = 0.0;

      for (unsigned int q = 0; q < n_q; ++q)
        {
          // We need to compute the forcing term at the current time (tn+1) and
          // at the old time (tn). deal.II Functions can be computed at a
          // specific time by calling their set_time method.

          // // Compute f(tn+1)
          // forcing_term.set_time(time);
          // const double f_new_loc =
          //   forcing_term.value(fe_values.quadrature_point(q));

          // // Compute f(tn)
          // forcing_term.set_time(time - deltat);
          // const double f_old_loc =
          //   forcing_term.value(fe_values.quadrature_point(q));

          const double alpha_loc = alpha.value(fe_values.quadrature_point(q));

          for (unsigned int i = 0; i < dofs_per_cell; ++i)
            {
              // cell_rhs(i) += (theta * f_new_loc + (1.0 - theta) * f_old_loc) *
              //                fe_values.shape_value(i, q) * fe_values.JxW(q);
              
              // (1/deltat + alpha) * integral(psi_i*psi_j)
              for (unsigned int j = 0; j < dofs_per_cell; ++j) {
                cell_rhs_matrix(i,j) += (1/deltat + alpha_loc) *
                  fe_values->shape_value(i, q) *
                  fe_values->shape_value(j,q) *
                  fe_values->JxW(q);
              }
            }
        }

      cell->get_dof_indices(dof_indices);
      // system_rhs.add(dof_indices, cell_rhs);
      rhs_matrix.add(dof_indices, cell_rhs_matrix);
    }

  // system_rhs.compress(VectorOperation::add);
  rhs_matrix.compress(VectorOperation::add);

  // Add the term that comes from the old solution.
  //rhs_matrix.vmult_add(system_rhs, solution_owned);

  // IMPORTANT: creation of the right hand side
  rhs_matrix.vmult(system_rhs, solution_owned);
}

template <int dim>
void
LinearFisherKolmogorov<dim>::solve_time_step()
{
  SolverControl solver_control(1000, 1e-6 * system_rhs.l2_norm());

  SolverCG<TrilinosWrappers::MPI::Vector> solver(solver_control);
  TrilinosWrappers::PreconditionSSOR      preconditioner;
  preconditioner.initialize(
    lhs_matrix, TrilinosWrappers::PreconditionSSOR::AdditionalData(1.0));

  solver.solve(lhs_matrix, solution_owned, system_rhs, preconditioner);
  pcout << "  " << solver_control.last_step() << " CG iterations" << std::endl;

  solution = solution_owned;
}

template <int dim>
void
LinearFisherKolmogorov<dim>::output(const unsigned int &time_step) const
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
LinearFisherKolmogorov<dim>::solve()
{

  pcout << "===============================================" << std::endl;

  time = 0.0;

  // Apply the initial condition.
  {
    pcout << "Applying the initial condition" << std::endl;

    exact_solution.set_time(time);
    VectorTools::interpolate(dof_handler, exact_solution, solution_owned);
    solution = solution_owned;

    // Output the initial solution.
    output(0);
    pcout << "-----------------------------------------------" << std::endl;
  }

  unsigned int time_step = 0;

  while (time < T - 0.5 * deltat)
    {
      time += deltat;
      ++time_step;

      pcout << "n = " << std::setw(3) << time_step << ", t = " << std::setw(5)
            << time << ":" << std::flush;

      assemble_lhs_matrix();
      assemble_rhs(time);
      solve_time_step();
      output(time_step);
    }
}

template <int dim>
double
LinearFisherKolmogorov<dim>::compute_error(const VectorTools::NormType &norm_type)
{
  FE_SimplexP<dim> fe_linear(1);
  MappingFE        mapping(fe_linear);

  const QGaussSimplex<dim> quadrature_error = QGaussSimplex<dim>(r + 2);

  exact_solution.set_time(time);

  Vector<double> error_per_cell;
  VectorTools::integrate_difference(mapping,
                                    dof_handler,
                                    solution,
                                    exact_solution,
                                    error_per_cell,
                                    quadrature_error,
                                    norm_type);

  const double error =
    VectorTools::compute_global_error(mesh, error_per_cell, norm_type);

  return error;
}