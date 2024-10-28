def ns_to_ms(nanoseconds: str | int):
    if isinstance(nanoseconds, str):
        nanoseconds = int(nanoseconds)
    return nanoseconds / 1000 / 1000 / 1000
