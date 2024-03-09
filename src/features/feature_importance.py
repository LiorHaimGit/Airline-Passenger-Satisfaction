# feature_importance.py

from sklearn.tree import DecisionTreeClassifier

def print_feature_importance(data):
    clf = DecisionTreeClassifier(max_depth=5, min_samples_split=5)
    clf.fit(data.drop('target', axis=1), data['target'])

    # Get feature importance
    importances = clf.feature_importances_
    feature_names = data.drop('target', axis=1).columns
    importance_list = list(zip(importances, feature_names))
    importance_list.sort(reverse=True)

    # Print the sorted list
    for importance, feature_name in importance_list:
        print(f"{feature_name}: {importance}")
def feature_importance(data):
    # print_feature_importance(data)

    # Drop less important features
    features_to_drop = ['Age',
                        'Seat comfort',
                        'On-board service',
                        'Leg room service',
                        'Inflight service',
                        'Gender',
                        'Food and drink',
                        'Flight Distance',
                        'Departure/Arrival time convenient',
                        'Departure Delay in Minutes',
                        'Arrival Delay in Minutes',
                        'Baggage handling']
    data = data.drop(columns=features_to_drop)

    return data
