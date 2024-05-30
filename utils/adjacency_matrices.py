import torch

# NAMING CONVENTION
# n{num nodes}_{num_non_zero_entries_first_row}...{num_non_zero_entries_last_row}

n2_22 = torch.tensor([[3/4, 1/4],
                      [1/4, 3/4]], dtype=torch.float)

n3_222 = torch.tensor([[2/3, 1/3, 0],
                       [0, 2/3, 1/3],
                       [1/3, 0, 2/3]], dtype=torch.float)

n3_111 = torch.tensor([[0, 1, 0],
                       [0, 0, 1],
                       [1, 0, 0]], dtype=torch.float)

n3_232 = torch.tensor([[1/2, 0, 1/2],
                       [1/4, 1/2, 1/4],
                       [1/2, 0, 1/2]], dtype=torch.float)

n4_2433 = torch.tensor([[1/2, 1/2, 0, 0],
                        [1/6, 1/2, 1/6, 1/6],
                        [0, 1/4, 1/2, 1/4],
                        [0, 1/6, 1/6, 2/3]], dtype=torch.float)
