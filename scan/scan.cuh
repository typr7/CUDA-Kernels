#pragma once

#include <cstddef>


size_t get_workspace_size(int num_elems);

void inclusive_scan(const int* in, int* out, void* workspace, int num_elems);

void exclusive_scan(const int* in, int* out, void* workspace, int num_elems);