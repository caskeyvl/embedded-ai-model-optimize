/* syscalls.c — Minimal newlib platform stubs for bare-metal STM32F407 */

#include <sys/stat.h>
#include <sys/time.h>
#include <errno.h>
#include <stdint.h>
#include <stddef.h>
#include <stdarg.h>
#include <reent.h>

/* ── Heap allocator ──────────────────────────────────────────────────────── */
extern uint32_t _ebss;
static uint8_t *heap_end = NULL;

void *_sbrk(int incr) {
    if (heap_end == NULL)
        heap_end = (uint8_t *)&_ebss;
    uint8_t *prev = heap_end;
    heap_end += incr;
    return (void *)prev;
}

/* ── Newlib reentrant structure init ─────────────────────────────────────── */
/*
 * Called from Reset_Handler before __libc_init_array.
 * newlib-nano's snprintf uses _impure_ptr internally; this initializes it.
 * ExecuTorch calls snprintf() in Method::resolve_operator(), so this must
 * run before main() calls load_method() (see README.md Gotcha 5).
 */
void init_newlib(void) {
    _REENT_INIT_PTR(_impure_ptr);
}

/* ── Byte-safe memcmp/memcpy/memset ──────────────────────────────────────── */
/*
 * newlib's word-wide implementations fault on unaligned accesses that
 * ExecuTorch's FlatBuffer parsing triggers (see README.md Gotcha 6).
 */
int memcmp(const void *s1, const void *s2, size_t n) {
    const unsigned char *p1 = (const unsigned char *)s1;
    const unsigned char *p2 = (const unsigned char *)s2;
    while (n--) {
        if (*p1 != *p2) return (int)*p1 - (int)*p2;
        p1++;
        p2++;
    }
    return 0;
}

void *memcpy(void *dst, const void *src, size_t n) {
    unsigned char *d = (unsigned char *)dst;
    const unsigned char *s = (const unsigned char *)src;
    while (n--) *d++ = *s++;
    return dst;
}

void *memset(void *s, int c, size_t n) {
    unsigned char *p = (unsigned char *)s;
    while (n--) *p++ = (unsigned char)c;
    return s;
}

/* ── snprintf / vsnprintf overrides ──────────────────────────────────────── */
/*
 * newlib's snprintf requires FILE* stream infrastructure that doesn't exist
 * on bare-metal (see README.md Gotcha 16). ExecuTorch uses snprintf() to
 * build operator name strings and error messages. This is not a complete
 * printf — unsupported specifiers pass through literally.
 */
int snprintf(char *buf, size_t size, const char *fmt, ...) {
    if (buf == NULL || size == 0) return 0;

    va_list args;
    va_start(args, fmt);

    size_t pos = 0;
    size_t remaining = size - 1;
    const char *f = fmt;

    while (*f && remaining > 0) {
        if (*f != '%') {
            buf[pos++] = *f++;
            remaining--;
            continue;
        }
        f++; /* skip '%' */

        /* Optional flags */
        int zero_pad = 0;
        if (*f == '0') { zero_pad = 1; f++; }

        /* Optional width */
        int width = 0;
        while (*f >= '0' && *f <= '9') { width = width * 10 + (*f++ - '0'); }

        /* Optional length modifier */
        int is_long = 0;
        if (*f == 'l') { is_long = 1; f++; }
        if (*f == 'z') { f++; } /* size_t — treat as unsigned long */

        char tmp[24];
        int tlen = 0;

        switch (*f) {
            case 's': {
                const char *s = va_arg(args, const char *);
                if (!s) s = "(null)";
                while (*s && remaining > 0) { buf[pos++] = *s++; remaining--; }
                break;
            }
            case 'c': {
                if (remaining > 0) { buf[pos++] = (char)va_arg(args, int); remaining--; }
                break;
            }
            case 'd': {
                long val = is_long ? va_arg(args, long) : (long)va_arg(args, int);
                if (val < 0 && remaining > 0) { buf[pos++] = '-'; remaining--; val = -val; }
                do { tmp[tlen++] = '0' + (int)(val % 10); val /= 10; } while (val && tlen < 23);
                if (zero_pad) while (tlen < width && tlen < 23) tmp[tlen++] = '0';
                while (tlen-- > 0 && remaining > 0) { buf[pos++] = tmp[tlen]; remaining--; }
                break;
            }
            case 'u': {
                unsigned long val = is_long ? va_arg(args, unsigned long)
                                            : (unsigned long)va_arg(args, unsigned int);
                do { tmp[tlen++] = '0' + (int)(val % 10); val /= 10; } while (val && tlen < 23);
                if (zero_pad) while (tlen < width && tlen < 23) tmp[tlen++] = '0';
                while (tlen-- > 0 && remaining > 0) { buf[pos++] = tmp[tlen]; remaining--; }
                break;
            }
            case 'x': case 'X': {
                unsigned long val = is_long ? va_arg(args, unsigned long)
                                            : (unsigned long)va_arg(args, unsigned int);
                const char *hex = (*f == 'x') ? "0123456789abcdef" : "0123456789ABCDEF";
                do { tmp[tlen++] = hex[val & 0xF]; val >>= 4; } while (val && tlen < 23);
                if (zero_pad) while (tlen < width && tlen < 23) tmp[tlen++] = '0';
                while (tlen-- > 0 && remaining > 0) { buf[pos++] = tmp[tlen]; remaining--; }
                break;
            }
            case 'p': {
                unsigned long val = (unsigned long)va_arg(args, void *);
                buf[pos++] = '0'; if (remaining) { buf[pos++] = 'x'; remaining--; }
                do { tmp[tlen++] = "0123456789abcdef"[val & 0xF]; val >>= 4; } while (val && tlen < 23);
                while (tlen-- > 0 && remaining > 0) { buf[pos++] = tmp[tlen]; remaining--; }
                break;
            }
            case '%':
                if (remaining > 0) { buf[pos++] = '%'; remaining--; }
                break;
            default:
                if (remaining > 0) { buf[pos++] = '%'; remaining--; }
                if (remaining > 0) { buf[pos++] = *f;  remaining--; }
                break;
        }
        f++;
    }

    buf[pos] = '\0';
    va_end(args);
    return (int)pos;
}

int vsnprintf(char *buf, size_t size, const char *fmt, va_list ap) {
    if (buf == NULL || size == 0) return 0;
    size_t pos = 0, remaining = size - 1;
    const char *f = fmt;
    while (*f && remaining > 0) {
        if (*f == '%' && *(f + 1) == 's') {
            const char *s = va_arg(ap, const char *);
            if (!s) s = "(null)";
            while (*s && remaining > 0) { buf[pos++] = *s++; remaining--; }
            f += 2;
        } else if (*f == '%' && *(f + 1) == 'd') {
            int val = va_arg(ap, int);
            char tmp[12]; int tlen = 0;
            if (val < 0 && remaining > 0) { buf[pos++] = '-'; remaining--; val = -val; }
            do { tmp[tlen++] = '0' + (val % 10); val /= 10; } while (val);
            while (tlen-- > 0 && remaining > 0) { buf[pos++] = tmp[tlen]; remaining--; }
            f += 2;
        } else {
            buf[pos++] = *f++; remaining--;
        }
    }
    buf[pos] = '\0';
    return (int)pos;
}

/* ── ExecuTorch PAL logging stub ──────────────────────────────────────────── */
/*
 * The default et_pal_emit_log_message() uses fprintf(stderr), which
 * requires FILE* infrastructure that crashes on bare-metal. -DET_LOG_ENABLED=0
 * doesn't suppress every call site, so a no-op override is still required
 * (see README.md Gotcha 8).
 */
void et_pal_emit_log_message(
        long timestamp, int level,
        const char *filename, const char *function, unsigned long line,
        const char *message, unsigned long length) {
    (void)timestamp; (void)level;    (void)filename;
    (void)function;  (void)line;     (void)message; (void)length;
}

/* ── POSIX syscall stubs ─────────────────────────────────────────────────── */
int _write(int fd, char *buf, int len)      { (void)fd; (void)buf; return len; }
int _read(int fd, char *buf, int len)       { (void)fd; (void)buf; (void)len; return 0; }
int _close(int fd)                          { (void)fd; return -1; }
int _isatty(int fd)                         { (void)fd; return 1; }
int _lseek(int fd, int off, int w)          { (void)fd; (void)off; (void)w; return 0; }
void _exit(int s)                           { (void)s; while (1); }
int _kill(int p, int s)                     { (void)p; (void)s; return -1; }
int _getpid(void)                           { return 1; }

int _fstat(int fd, struct stat *st) {
    (void)fd;
    st->st_mode = S_IFCHR;
    return 0;
}

int _gettimeofday(struct timeval *tv, void *tz) {
    (void)tz;
    if (tv) { tv->tv_sec = 0; tv->tv_usec = 0; }
    return 0;
}
