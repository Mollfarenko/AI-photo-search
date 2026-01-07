def build_where_clause(
    year: int | None = None,
    month: int | None = None,
    time_of_day: str | None = None,
    camera_make: str | None = None,
    camera_model: str | None = None,
):
    where = {}

    if year is not None:
        where["year"] = year

    if month is not None:
        where["month"] = month

    if time_of_day:
        where["period_of_day"] = time_of_day

    if camera_make:
        where["camera_make"] = camera_make

    if camera_model:
        where["camera_model"] = camera_model

    return where
