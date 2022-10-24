import cv2
import numpy as np
import copy
EPSILON = 1e-8


def norm2(x, axis=None):
    return np.sqrt(np.sum(x ** 2, axis=axis))


def cos(p1, p2):
    return (p1 * p2).sum() / (norm2(p1) * norm2(p2) + EPSILON)


class PolyBBox(object):

    def __init__(self, pts, n_parts=16, num_width_line=5):
        self.pts = self.remove_points(pts)
        self.pts_num = len(self.pts)
        self.n_parts = n_parts
        self.num_width_line = num_width_line

        self.bbox = (np.min(self.pts[:, 0]), np.min(self.pts[:, 1]),
                     np.max(self.pts[:, 0]), np.max(self.pts[:, 1]))
        self.cx, self.cy = (self.bbox[0] + self.bbox[2]) / 2, (self.bbox[1] + self.bbox[3]) / 2
        self.height, self.width = (self.bbox[3] - self.bbox[1]), (self.bbox[2] - self.bbox[0])

    def remove_points(self, pts):
        ''' 
        remove point if area is almost unchanged after removing it
        '''
        rm_pts_idx = []
        if len(pts) > 4:
            ori_area = cv2.contourArea(pts)
            for i in range(len(pts)):
                # attempt to remove pts[i]
                index = list(range(len(pts)))
                index.remove(i)
                area = cv2.contourArea(pts[index])
                if np.abs(ori_area - area) / ori_area < 0.017 and len(pts) - len(rm_pts_idx) > 4:
                    rm_pts_idx.append(i)
        return np.array([pt for i, pt in enumerate(pts) if i not in rm_pts_idx])

    def find_short_sides(self):
        if self.pts_num > 4:
            points = np.concatenate([self.pts, self.pts[:3]])
            candidate = []
            for i in range(1, self.pts_num + 1):
                prev_edge = points[i] - points[i - 1]
                next_edge = points[i + 2] - points[i + 1]
                if cos(prev_edge, next_edge) < -0.8:
                    candidate.append((i % self.pts_num, (i + 1) % self.pts_num, norm2(points[i] - points[i + 1])))

            if len(candidate) != 2 or candidate[0][0] == candidate[1][1] or candidate[0][1] == candidate[1][0]:
                # if candidate number < 2, or two bottom are joined, select 2 farthest edge
                mid_list = []
                for i in range(self.pts_num):
                    mid_point = (points[i] + points[(i + 1) % self.pts_num]) / 2
                    mid_list.append((i, (i + 1) % self.pts_num, mid_point))

                dist_list = []
                for i in range(self.pts_num):
                    for j in range(self.pts_num):
                        s1, e1, mid1 = mid_list[i]
                        s2, e2, mid2 = mid_list[j]
                        dist = norm2(mid1 - mid2)
                        dist_list.append((s1, e1, s2, e2, dist))
                short_sides_idx = np.argsort([dist for s1, e1, s2, e2, dist in dist_list])[-2:]
                short_sides = [dist_list[short_sides_idx[0]][:2], dist_list[short_sides_idx[1]][:2]]
                if short_sides[0][0] == short_sides[1][1] or short_sides[0][1] == short_sides[1][0]:
                    short_sides_idx = np.argsort([dist for s1, e1, s2, e2, dist in dist_list])[-1:]
                    short_sides = [dist_list[short_sides_idx[0]][0:2], dist_list[short_sides_idx[0]][2:4]]
            else:
                short_sides = [candidate[0][:2], candidate[1][:2]]

        else:
            d1 = norm2(self.pts[1] - self.pts[0]) + norm2(self.pts[2] - self.pts[3])
            d2 = norm2(self.pts[2] - self.pts[1]) + norm2(self.pts[0] - self.pts[3])
            short_sides = [(0, 1), (2, 3)] if d1 < d2 else [(1, 2), (3, 0)]
        assert len(short_sides) == 2, 'more or less than 2 short sides'
        return short_sides

    def find_long_sides(self, short_sides):
        ss1_start, ss1_end = short_sides[0]
        ss2_start, ss2_end = short_sides[1]

        long_sides_list = [[], []]
        i = (ss1_end + 1) % self.pts_num
        while (i != ss2_end):
            long_sides_list[0].append(((i - 1) % self.pts_num, i))
            i = (i + 1) % self.pts_num

        i = (ss2_end + 1) % self.pts_num
        while (i != ss1_end):
            long_sides_list[1].append(((i - 1) % self.pts_num, i))
            i = (i + 1) % self.pts_num

        return long_sides_list

    def partition_long_sides(self):
        '''
        cover text region with several parts
        :return:
        '''

        self.short_sides = self.find_short_sides()  # find two short sides of this Text
        self.long_sides_list = self.find_long_sides(self.short_sides)  # find two long sides sequence

        inner_pts1 = self.split_side_seqence(self.long_sides_list[0])
        inner_pts2 = self.split_side_seqence(self.long_sides_list[1])
        inner_pts2 = inner_pts2[::-1]  # innverse one of long edge

        center_pts = (inner_pts1 + inner_pts2) / 2  # disk center

        return inner_pts1, inner_pts2, center_pts  # , radii

    def split_side_seqence(self, long_sides):
        side_lengths = [norm2(self.pts[v1] - self.pts[v2]) for v1, v2 in long_sides]
        sides_cumsum = np.cumsum([0] + side_lengths)
        total_length = sum(side_lengths)
        length_per_part = total_length / self.n_parts

        cur_node = 0  # first point
        mid_pt_list = []

        for i in range(1, self.n_parts):
            curr_len = i * length_per_part

            while (curr_len > sides_cumsum[cur_node + 1]):
                cur_node += 1

            v1, v2 = long_sides[cur_node]
            pt1, pt2 = self.pts[v1], self.pts[v2]

            # start_point = self.pts[long_edge[cur_node]]
            end_shift = curr_len - sides_cumsum[cur_node]
            ratio = end_shift / side_lengths[cur_node]
            new_pt = pt1 + ratio * (pt2 - pt1)
            # print(cur_end, point_cumsum[cur_node], end_shift, edge_length[cur_node], '=', new_point)
            mid_pt_list.append(new_pt)

        # add first and last point
        pt_first = self.pts[long_sides[0][0]]
        pt_last = self.pts[long_sides[-1][1]]
        mid_pt_list = [pt_first] + mid_pt_list + [pt_last]
        return np.stack(mid_pt_list)

    def gen_poly_label(self):
        if self.pts_num < 4:
            print('WARNING: number of points is {} less than 4.\n'.format(self.pts_num))
            print(self.pts)
            return None

        self.short_sides = self.find_short_sides()  # find two short sides of this Text
        self.long_sides_list = self.find_long_sides(self.short_sides)  # find two long sides sequence

        if len(self.long_sides_list[0]) == 0 or len(self.long_sides_list[1]) == 0:
            print('WARNING: short and long sides {}\t{}.'.format(self.short_sides, self.long_sides_list))
            return None

        side_pts1 = self.split_side_seqence(self.long_sides_list[0])
        side_pts2 = self.split_side_seqence(self.long_sides_list[1])
        side_pts2 = side_pts2[::-1]  # innverse one of long edge
        center_pts = (side_pts1 + side_pts2) / 2  # disk center

        orientation_angle = np.divide(center_pts[0][1] - center_pts[-1][1], center_pts[0][0] - center_pts[-1][0])
        if orientation_angle > 1.19 or orientation_angle < -1.19:
            direction = 1
        else:
            direction = 0
        if direction == 0 and center_pts[-1, 0] > center_pts[0, 0]:
            side_pts1 = side_pts1[::-1]
            side_pts2 = side_pts2[::-1]
            center_pts = center_pts[::-1]
        elif direction == 1 and center_pts[-1, 1] > center_pts[0, 1]:
            side_pts1 = side_pts1[::-1]
            side_pts2 = side_pts2[::-1]
            center_pts = center_pts[::-1]

        if direction == 0 and side_pts1[side_pts1.shape[0]//2,1] > side_pts2[side_pts1.shape[0]//2,1]:
            inverse_flag = True
        elif direction == 1 and side_pts1[side_pts1.shape[0]//2,0] < side_pts2[side_pts1.shape[0]//2,0]:
            inverse_flag = True
        else:
            inverse_flag = False
        if inverse_flag:
            line_pts = np.array(
                [[side_pts2[i], center_pts[i], side_pts1[i]] for i in np.linspace(0, self.n_parts, num=self.num_width_line, dtype=np.int32)])
        else:
            line_pts = np.array(
                [[side_pts1[i], center_pts[i], side_pts2[i]] for i in np.linspace(0, self.n_parts, num=self.num_width_line, dtype=np.int32)])

        origin_line_pts = copy.deepcopy(line_pts)
        origin_poly_pts = origin_line_pts[:,[0,2],:]

        # Normalization
        center_pts[:, 0] -= self.cx
        center_pts[:, 1] -= self.cy
        center_pts[:, 0] /= self.width
        center_pts[:, 1] /= self.height

        line_pts[:, :, 0] -= self.cx
        line_pts[:, :, 1] -= self.cy
        line_pts[:, :, 0] /= self.width
        line_pts[:, :, 1] /= self.height

        delta_x = line_pts[:, 0, 0] - line_pts[:, 2, 0]
        delta_y = line_pts[:, 0, 1] - line_pts[:, 2, 1]

        if direction == 0:
            line_angle = np.arctan(np.divide(delta_x, delta_y))
        elif direction == 1:
            line_angle = np.arctan(np.divide(delta_y, delta_x))
        else:
            raise NotImplementedError

        line_length = np.array([line_pts[:, 0, :] - line_pts[:, 2, :]]) ** 2
        line_length = np.squeeze(line_length)
        line_length = np.sum(line_length, axis=1) ** (0.5)

        line_x = line_pts[:, 1, 0]
        line_y = line_pts[:, 1, 1]

        origin_poly_pts = np.concatenate((origin_poly_pts[:,0,:],origin_poly_pts[:,-1,:]),axis=0).astype(np.int)
        origin_poly_pts = cv2.boxPoints(cv2.minAreaRect(origin_poly_pts))
        origin_poly_pts =np.stack((origin_poly_pts[1:3,:],origin_poly_pts[[0,-1],:]),axis=0)
        label_list = [center_pts, line_angle, line_length, line_x, line_y, direction, origin_poly_pts]
        if any(map(lambda x: np.isnan(x).any(), label_list)):
            print('WARNING: poly label contains nan.')
            # print(label_list)
            return None

        poly_label = dict(
            poly=self.pts,
            bbox=self.bbox,
            center_line=center_pts,
            line_angle=line_angle,
            line_length=line_length,
            line_x=line_x,
            line_y=line_y,
            direction=np.array([direction]),
            origin_poly_pts=origin_poly_pts
        )

        return poly_label


