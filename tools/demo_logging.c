#include <stdio.h>
#include <stdlib.h>
#include "niyah_timer.h"
#include "niyah_ppl.h"

/* Formatting demonstration only: every loss/throughput metric is synthetic. */
int main(void) {
    FILE *log = NULL;
#ifdef _MSC_VER
    if (fopen_s(&log, "synthetic_demo_log.csv", "wx") != 0) return 1;
#else
    log = fopen("synthetic_demo_log.csv", "wx");
#endif
    if (!log) return 1;
    fprintf(stderr, "SYNTHETIC DEMO: no training or heldout evaluation is performed.\n");
    fprintf(log, "synthetic_step,synthetic_loss,synthetic_lr,synthetic_tokens_per_sec,synthetic_val_loss,synthetic_val_ppl\n");
    const int batch_tokens = 4096, log_every = 10, val_every = 50;
    double t0 = niyah_now_sec();
    long tok = 0;
    for (int step = 1; step <= 100; step++) {
        double loss = 5.0 / (1.0 + 0.05 * step);
        tok += batch_tokens;
        if (step % log_every == 0) {
            double dt = niyah_now_sec() - t0;
            fprintf(log, "%d,%.6f,%.8f,%.1f,,\n", step, loss, 3e-4, tok / dt);
            fflush(log);
            t0 = niyah_now_sec(); tok = 0;
        }
        if (step % val_every == 0) {
            niyah_ppl_acc a; niyah_ppl_reset(&a);
            for (int b = 0; b < 5; b++) niyah_ppl_add(&a, loss + 0.1, batch_tokens);
            fprintf(log, "%d,,,,%.6f,%.3f\n", step, niyah_ppl_val_loss(&a), niyah_ppl_value(&a));
            fflush(log);
        }
    }
    {
        const int failed = ferror(log);
        const int close_failed = fclose(log) != 0;
        return failed || close_failed ? 1 : 0;
    }
}
