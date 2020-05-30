import numpy


class FrameStacker:

    @staticmethod
    def stack_last_frame(frames: numpy.ndarray, nb_to_stack: int, stack_in_last_component: bool) -> numpy.ndarray:
        """Returns the last frame preceded by the nb_to_stack - 1 previous ones. If stack_in_last_component is False,
        we return a numpy array of shape   original frame shape x nb_to_stack. Otherwise, the returned array will have
        shape   original frame shape[0:-1] x (original frame shape[-1] * nb_to_stack). E.g. when the frames represent
        color images with three color channels, the returned frames will have 3*nb_to_stack channels, with the first
        three being the original ones.
        If there are fewer frames in frames than nb_to_stack, the last frame will be repeated.
        """
        stack = None
        total_nb_frames = frames.shape[0]
        frame_nb = total_nb_frames - nb_to_stack
        for _ in range(nb_to_stack):
            if frame_nb < 0:
                frame_nb = 0
            if frame_nb >= total_nb_frames:
                frame_nb = total_nb_frames - 1  # if necessary (at the start), repeat the current frame
            frame = frames[frame_nb]
            frame_nb += 1

            if stack is None:
                if stack_in_last_component:
                    stack = frame
                else:
                    stack = frame.reshape(frame.shape + (1,))
            else:
                if stack_in_last_component:
                    stack = numpy.concatenate((stack, frame), axis=len(frame.shape) - 1)
                else:
                    stack = numpy.concatenate((stack, frame.reshape(frame.shape + (1,))), axis=len(frame.shape))

        return stack

    @staticmethod
    def stack_all_frames(frames: numpy.ndarray, nb_to_stack: int, stack_in_last_component: bool) -> numpy.ndarray:
        """Returns a numpy array of stacked frames, where every stacked frame consists of that base frame preceded
        by the nb_to_stack - 1 previous frames. If stack_in_last_component is False, we will then return an array
        with shape (<number of frames> x <frame shape> x nb_to_stack). E.g. if we get a batch of 32 images of size
        210 x 160 x 3, and we want to always stack 5, the returned array's shape is 32 x 210 x 160 x 3 x 5.
        By contrast, if stack_in_last_component is True, then we will not create an extra dimension, but stack in
        the last component. In our example we would then return an array of shape 32 x 210 x 160 x 15.
        For the first nb_to_stack - 1 frames in frames the base frame will be repeated (as there are not enough previous
        frames).
        """
        stack = []
        for i in range(frames.shape[0]):
            stack.append(FrameStacker.stack_last_frame(frames[0:i+1], nb_to_stack, stack_in_last_component))

        stack = numpy.array(stack)

        """
        Note: this is actually faster than copying whole subarrays, as below. Especially for the case when we
        stack in the last component.
        (But below we added a component in second place, instead of the last, so transposing would be necessary.)
        """
        """      
        stack = numpy.empty((frames.shape[0], nb_to_stack) + frames.shape[1:])
        """
        """     
        e.g. nb_to_stack = 5
         j: 0 1 2 3 4 5 6 ... L-1
        i    
        0:  0 0 0 0|0 1 2 ... L-5
        1:  0 1 1 1|1 2 3 ... L-4
        2:  0 1 2 2|2 3 4 ... L-3
        3:  0 1 2 3|3 4 5 ... L-2
        4:  0 1 2 3|4 5 6 ... L-1
        """

        """
        for i in range(nb_to_stack):
            # right of | above:
            stack[nb_to_stack - 1:, i] = frames[i:frames.shape[0] - nb_to_stack + i + 1]  # + 1: exclusive
            # left of |:
            for j in range(nb_to_stack - 2):
                stack[j, i] = frames[min(i, j)]
                # i.e. add frames[j] until we get to i, then keep adding frames[i] until we're full
        """
        """
        if stack_in_last_component:
            # In the example from the documentation:
            # 1) Swap first two axes: 32 x 5 x 210 x 160 x 3 -> 5 x 32 x 210 x 160 x 3
            # 2) Concatenate array to the last component:    ->     32 x 210 x 160 x 15
            return numpy.concatenate(numpy.swapaxes(stack, 0, 1), axis=len(stack.shape) - 2)
        else:
        """
        return stack
