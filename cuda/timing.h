/*
 * timing.h
 *
 *  Created on: 2 nov. 2016
 *      Author: raul
 */

#ifndef TIMING_H_
#define TIMING_H_

#include <sys/time.h>

#ifdef __cplusplus
extern "C" {
#endif

double getElapsedUsec(struct timeval* a, struct timeval* b);
double getElapsedMsec(struct timeval* a, struct timeval* b);
double getElapsedSec(struct timeval* a, struct timeval* b);

#ifdef __cplusplus
}
#endif

#endif /* TIMING_H_ */
