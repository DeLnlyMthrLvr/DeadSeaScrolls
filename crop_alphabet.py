from matplotlib import pyplot as plt
import numpy as np
from alphabet import load_alphabet
import imageio


def create_cropped_alphabet():
    a = load_alphabet(include_paths=True)
    modified = dict()

    for char, images in a.items():
        example_shown = 0
        example_l = 0

        modified[char] = []

        for i_img, (img, path) in enumerate(images):
            name = path.name
            new_path = path.parent.parent.parent / "alphabet_cropped" / char / name
            new_path.parent.mkdir(exist_ok=True)

            modified[char].append([img, new_path])

            T = 5
            mask = img < 200
            height, _ = img.shape

        # The y coordinates of the "ink molecules"
            indices = np.where(mask)[0]
            cog = int(np.mean(indices))

            counts = np.sum(mask, axis=1)
            empty = counts < 2

            if empty[cog]:
                print(char, i_img + 1, height)
                fig, ax = plt.subplots()
                ax.imshow(img)
                ax.axhline(cog)

            assert not empty[cog]


            change = np.diff(empty)
            change_i = np.where(change)[0].tolist()

            change_i.append(height)
            if change_i[0] != 0:
                change_i.insert(0, 0)

            main_bounds_i = [None, None]
            i = 0
            for prev, fol in zip(change_i[:-1], change_i[1:], strict=True):
                if prev <= cog <= fol:
                    main_bounds_i[0] = i
                    main_bounds_i[1] = i + 1
                    break

                i += 1


            assert main_bounds_i[0] is not None

        # Travel up
            l = main_bounds_i[0]
            cut_from_top = 0
            land = False
            while l > 0:
                if not land:
                    distance = change_i[l] - change_i[l - 1]

                    if distance > T:
                        cut_from_top = change_i[l] - 1
                        break

                l -= 1
                land = not land

        # Travel down
            t = main_bounds_i[1]
            maintain_top = height
            land = False
            while t < (len(change_i) - 1):
                if not land:
                    distance = change_i[t + 1] - change_i[t]

                    if distance > T:
                        maintain_top = change_i[t] + 2
                        break

                t += 1
                land = not land

            if cut_from_top > 0 or maintain_top < height:
                cropped = img[cut_from_top:maintain_top, :]
                modified[char][-1][0] = cropped

                if example_shown < example_l:
                    fig, ax = plt.subplots(1, 2)
                    ax[0].imshow(img)

                    ax[1].imshow(cropped)

                    fig.suptitle(char)

                    example_shown += 1

        # p = counts / counts.sum()
        # y = np.arange(counts.size)
        # mean = np.sum(y * p)

        # plt.bar(y, counts)
        # plt.axvline(mean, color="red")


    for images in modified.values():
        for img, path in images:
            imageio.imwrite(path, img)


if __name__ == "__main__":
    create_cropped_alphabet()