/*
 * time.c
 *
 *  Created on: 2 nov. 2016
 *      Author: raul
 */


#include "timing.h"



#ifdef __cplusplus
extern "C" {
#endif

double getElapsedUsec(struct timeval* a, struct timeval* b) {
    return (b->tv_sec - a->tv_sec)*1e6 + (b->tv_usec - a->tv_usec);
}

double getElapsedMsec(struct timeval* a, struct timeval* b) {
    return getElapsedUsec(a,b)/1e3;
}

double getElapsedSec(struct timeval* a, struct timeval* b) {
    return getElapsedUsec(a,b)/1e6;
}

#ifdef __cplusplus
}
#endif
