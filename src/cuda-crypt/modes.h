/*
 * Copyright 2008-2016 The OpenSSL Project Authors. All Rights Reserved.
 *
 * Licensed under the OpenSSL license (the "License").  You may not use
 * this file except in compliance with the License.  You can obtain a copy
 * in the file LICENSE in the source distribution or at
 * https://www.openssl.org/source/license.html
 */

#ifndef HEADER_MODES_H
# define HEADER_MODES_H

# include <stddef.h>

# ifdef  __cplusplus
extern "C" {
# endif
typedef void (*block128_f) (const unsigned char in[16],
                            unsigned char out[16], const void *key);

typedef void (*cbc128_f) (const unsigned char *in, unsigned char *out,
                          size_t len, const void *key,
                          unsigned char ivec[16], int enc);

typedef void (*ccm128_f) (const unsigned char *in, unsigned char *out,
                          size_t blocks, const void *key,
                          const unsigned char ivec[16],
                          unsigned char cmac[16]);

__host__ __device__ void CRYPTO_cbc128_encrypt(const unsigned char *in, unsigned char *out,
                           uint32_t len, const void *key,
                           unsigned char* ivec, const uint32_t* Te3);

void CRYPTO_cbc128_decrypt(const unsigned char *in, unsigned char *out,
                           size_t len, const void *key,
                           unsigned char ivec[16], block128_f block);

# ifdef  __cplusplus
}
# endif

#endif
