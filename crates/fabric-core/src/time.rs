// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! RFC 3339 UTC timestamps with millisecond precision.
//!
//! Core has no date crate, so this converts Unix milliseconds to a civil date
//! with Howard Hinnant's `civil_from_days` algorithm.

use std::time::{SystemTime, UNIX_EPOCH};

const MILLIS_PER_DAY: i64 = 86_400_000;

/// Format `time` as RFC 3339 UTC with three fraction digits, such as
/// `2026-10-14T18:01:10.205Z`.
pub(crate) fn rfc3339_millis(time: SystemTime) -> String {
    format_unix_millis(unix_millis(time))
}

/// Milliseconds since the Unix epoch; negative before it.
fn unix_millis(time: SystemTime) -> i64 {
    match time.duration_since(UNIX_EPOCH) {
        Ok(after) => i64::try_from(after.as_millis()).unwrap_or(i64::MAX),
        Err(before) => -i64::try_from(before.duration().as_millis()).unwrap_or(i64::MAX),
    }
}

fn format_unix_millis(unix_millis: i64) -> String {
    let days = unix_millis.div_euclid(MILLIS_PER_DAY);
    let millis_of_day = unix_millis.rem_euclid(MILLIS_PER_DAY);
    let (year, month, day) = civil_from_days(days);
    let seconds_of_day = millis_of_day / 1000;
    format!(
        "{year:04}-{month:02}-{day:02}T{:02}:{:02}:{:02}.{:03}Z",
        seconds_of_day / 3600,
        seconds_of_day / 60 % 60,
        seconds_of_day % 60,
        millis_of_day % 1000,
    )
}

fn civil_from_days(days: i64) -> (i64, i64, i64) {
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let day_of_era = z.rem_euclid(146_097);
    let year_of_era =
        (day_of_era - day_of_era / 1460 + day_of_era / 36_524 - day_of_era / 146_096) / 365;
    let day_of_year = day_of_era - (365 * year_of_era + year_of_era / 4 - year_of_era / 100);
    let month_index = (5 * day_of_year + 2) / 153;
    let day = day_of_year - (153 * month_index + 2) / 5 + 1;
    let month = if month_index < 10 {
        month_index + 3
    } else {
        month_index - 9
    };
    let year = year_of_era + era * 400 + i64::from(month <= 2);
    (year, month, day)
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use super::*;

    #[test]
    fn rfc3339_millis_formats_fixed_vectors() {
        assert_eq!(format_unix_millis(0), "1970-01-01T00:00:00.000Z");
        assert_eq!(
            format_unix_millis(1_792_000_870_205),
            "2026-10-14T18:01:10.205Z"
        );
        assert_eq!(
            format_unix_millis(951_782_400_000),
            "2000-02-29T00:00:00.000Z"
        );
        assert_eq!(
            format_unix_millis(4_107_542_399_999),
            "2100-02-28T23:59:59.999Z"
        );
        assert_eq!(format_unix_millis(-1), "1969-12-31T23:59:59.999Z");
        assert_eq!(
            rfc3339_millis(UNIX_EPOCH + Duration::from_millis(1_792_000_870_205)),
            "2026-10-14T18:01:10.205Z"
        );
        assert_eq!(
            rfc3339_millis(UNIX_EPOCH - Duration::from_millis(1)),
            "1969-12-31T23:59:59.999Z"
        );
    }
}
