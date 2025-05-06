    print("Preference Dataset:")
    preference = preference.sort_values(by="uniq_id")
    print(preference.head())
    print(preference.info())