#pragma once

#include <cassert>
#include <cstdio>

#define HARDEN_ASSERT(expr)                                                    \
    if (!(expr)) {                                                             \
        puts("Assertion " #expr " failed at " __FILE__);                       \
        printf(":%d\n", __LINE__);                                             \
        abort();                                                               \
    }
