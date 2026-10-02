#!/usr/bin/env python3
# encoding: utf-8
# Inverse kinematics of 4DOF robotic arm: corresponding given coordinate (X, Y, Z) and pitch angle. Calculate the rotation angle of each joint.
# 2020/07/20 Aiden
import logging
from math import *

# CRITICAL, ERROR, WARNING, INFO, DEBUG
logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)

class IK:
    # Count the servo from the bottom to the top
    # common parameter. It refers to link parameter of 4DOF robotic arm
    l1 = 6.10    #the distance between robotic arm chassis center and NO.2 servo central shaft is 6.10 cm
    l2 = 10.16   #the distance between NO.2 servo and NO.3 servo is 10.16 cm
    l3 = 9.64    #the distance between NO.3 servo and NO.4 servo is 9.64cm
    l4 = 0.00    #no specific assignment is made here, but reassignment is made based on the selection made during initialization

    # parameter only for air pump version
    l5 = 4.70  #the distance between NO.4 servo and the top of suction nozzle is 4.70cm. 
    l6 = 4.46  #distance between the top of suction nozzle and suction nozzle is 4.46cm
    alpha = degrees(atan(l6 / l5))  #calculate the angle between 15 and 14

    def __init__(self, arm_type): #adapt the parameter according to different end-of-tool
        self.arm_type = arm_type
        if self.arm_type == 'pump': #if it is the robotic arm with sir pump.
            self.l4 = sqrt(pow(self.l5, 2) + pow(self.l6, 2))  #the fourth steering gear to the nozzle serves as the fourth link
        elif self.arm_type == 'arm':
            self.l4 = 16.65  #distance between NO.4 servo and the end of robotic arm, when the gripper is closed completely, is 16.6cm.  

    def setLinkLength(self, L1=l1, L2=l2, L3=l3, L4=l4, L5=l5, L6=l6):
        # change the link length of the robotic arm to adapt to robotic arm in same structure but different length.
        self.l1 = L1
        self.l2 = L2
        self.l3 = L3
        self.l4 = L4
        self.l5 = L5
        self.l6 = L6
        if self.arm_type == 'pump':
            self.l4 = sqrt(pow(self.l5, 2) + pow(self.l6, 2))
            self.alpha = degrees(atan(self.l6 / self.l5))

    def getLinkLength(self):
        # obtain the current link length
        if self.arm_type == 'pump':
            return {"L1":self.l1, "L2":self.l2, "L3":self.l3, "L4":self.l4, "L5":self.l5, "L6":self.l6}
        else:
            return {"L1":self.l1, "L2":self.l2, "L3":self.l3, "L4":self.l4}

    def getRotationAngle(self, coordinate_data, Alpha):
        # Given specific coordinate and pitch angle. Return rotation angle of each joint. If there is no solution, return False.
        # coordinate_data is coordinate of gripper end in cm. It is passed in as tuple, for example (0, 5, 10) 
        # Alpha is the angle between gripper and the horizontal plane, in degree.

        # the end of gripper is P (X, Y, Z), coordinate origin is 0, and the origin is the projection of the pan-tilt center on the ground, and the projection of point P on the ground is P_
        # The intersection of l1 and l2 is A, the intersection of l2 and l3 is B, and the intersection of l3 and l4 is C
        # CD is perpendicular to PD, CD is perpendicular to z-axis, then the pitch angle Alpha is the angle between DC and PC, AE is perpendicular to DP_, and E is on DP_, CF is perpendicular to AE, and F is on AE
        # For example, the angle between AB and BC is ABC
        X, Y, Z = coordinate_data
        if self.arm_type == 'pump':
            Alpha -= self.alpha
        #Find the rotation angle of the base 
        theta6 = degrees(atan2(Y, X))
 
        P_O = sqrt(X*X + Y*Y) #distance between P_ and origin 0 
        CD = self.l4 * cos(radians(Alpha))
        PD = self.l4 * sin(radians(Alpha)) #when the pitch angle is positive, PD is positive. When pitch angle is negative, PD is negative
        AF = P_O - CD
        CF = Z - self.l1 - PD
        AC = sqrt(pow(AF, 2) + pow(CF, 2))
        if round(CF, 4) < -self.l1:
            logger.debug('the height is less than 0, CF(%s)<l1(%s)', CF, -self.l1)
            return False
        if self.l2 + self.l3 < round(AC, 4): #The sum of both sides is less than the third side
            logger.debug('the link structure cannot be built, l2(%s) + l3(%s) < AC(%s)', self.l2, self.l3, AC)
            return False

        #Find theat4
        cos_ABC = round(-(pow(AC, 2)- pow(self.l2, 2) - pow(self.l3, 2))/(2*self.l2*self.l3), 4) #cosine theory
        if abs(cos_ABC) > 1:
            logger.debug('the link structure cannot be built, abs(cos_ABC(%s)) > 1', cos_ABC)
            return False
        ABC = acos(cos_ABC) #inverse trig calculates the radian
        theta4 = 180.0 - degrees(ABC)

        #Find theta5
        CAF = acos(AF / AC)
        cos_BAC = round((pow(AC, 2) + pow(self.l2, 2) - pow(self.l3, 2))/(2*self.l2*AC), 4) #cosine theory
        if abs(cos_BAC) > 1:
            logger.debug('the link structure cannot be built, abs(cos_BAC(%s)) > 1', cos_BAC)
            return False
        if CF < 0:
            zf_flag = -1
        else:
            zf_flag = 1
        theta5 = degrees(CAF * zf_flag + acos(cos_BAC))

        #Find theta3
        theta3 = Alpha - theta5 + theta4
        if self.arm_type == 'pump':
            theta3 += self.alpha

        return {"theta3":theta3, "theta4":theta4, "theta5":theta5, "theta6":theta6} # When there is solution, return angle dictionary
            
if __name__ == '__main__':
    ik = IK('arm')
    ik.setLinkLength(L1=ik.l1 + 0.89, L4=ik.l4 - 0.3)
    print('link length：', ik.getLinkLength())
    print(ik.getRotationAngle((0, 0, ik.l1 + ik.l2 + ik.l3 + ik.l4), 90))
