
def weather_risk(temp,humidity,rainfall):
    reasons=[]; score=0
    if humidity>=80: score+=2; reasons.append("High humidity may support fungal disease development")
    if 18<=temp<=30: score+=1; reasons.append("Temperature is within a common disease-favorable range")
    if rainfall>0: score+=2; reasons.append("Recent rainfall can increase leaf wetness")
    return ("Higher environmental disease pressure" if score>=4 else "Moderate environmental disease pressure" if score>=1 else "Lower environmental disease pressure", reasons)
